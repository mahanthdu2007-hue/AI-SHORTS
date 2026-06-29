import json
from typing import List, Optional
from pydantic import BaseModel, Field

from .models import Transcript, Highlight, ScoreCriteria, ShortMetadata
from .providers import BaseLLMProvider
from .logger import get_logger

logger = get_logger("highlights")


class HighlightList(BaseModel):
    highlights: List[Highlight]


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only."""

class ContentInfo(BaseModel):
    content_type: str = Field(description="The type of content (e.g. podcast, interview)")
    density: str = Field(description="low, medium, or high")


STAGE1_PROMPT = """You are a highly-paid viral video strategist producing for YouTube Shorts, TikTok, and Instagram Reels.
Your only goal is to maximize VIEWER RETENTION. Do not simply find interesting segments—find the highly-addictive, scroll-stopping moments that people cannot stop watching.

Content type: {content_type} | Density: {density}

OPTIMIZE EXCLUSIVELY FOR:
- Curiosity gaps (e.g. "The real reason you're poor is...")
- Emotional moments & extreme vulnerability
- Unexpected twists & shocking statistics
- Powerful, controversial, or polarizing opinions
- Life-changing advice & profound realizations
- High-stakes money, business, success, and failure stories
- Laugh-out-loud funny moments
- Captivating personal stories

STRICTLY AVOID & REJECT (DO NOT CLIP):
- Greetings ("Hey guys", "Welcome to the podcast")
- Small talk, chit-chat, and weak transitions
- Sponsor messages and ads
- Repeated ideas or rambling
- Anything that doesn't hook the viewer in the first 3 seconds
- Clips that start in the middle of a story or end before the payoff
- Monotone speech, heavy filler words (um, uh), long pauses, and repetitive explanations

RULES:
- Extract exactly 25 candidate clips. Prioritize diverse clips across the video.
- Find clips that form a COMPLETE mini-story. Look for narrative structure: Setup, Conflict, Rising Action, Peak, and Resolution.
- Read for transcript signals like `[laughs]`, `[cheers]`, exclamation marks (!), and ALL-CAPS (emphasis) to find high-energy moments.
- Duration sweet spot: 25-60 seconds. Do not cut a thought off prematurely.
- Provide a highly-clickable title and a punchy 3-second hook sentence.
- Identify start and end times accurately based on the transcript timestamps.
- Score is not evaluated in this pass, just provide 0.

Respond ONLY with valid JSON matching the schema.
"""


STAGE2_ELIMINATION_PROMPT = """You are a senior YouTube Shorts producer reviewing rough cuts.
I have a list of candidate highlights. Your task is to perform a strict elimination and refinement pass (Stage 2).

Your goals:
1. Merge overlapping clips into single, cohesive clips if they cover the same topic.
2. Remove weak clips entirely by setting `rejection_reason`. Reject if it's an Intro, Outro, Sponsor segment, Dead air, starts in the middle of a thought, or lacks a payoff.
3. You do not need to score them yet (leave score as 0).

Candidates:
{candidates_json}

Return the refined and filtered list of clips.
Respond ONLY with valid JSON matching the schema.
"""


STAGE3_SCORING_PROMPT = """You are a senior YouTube Shorts producer selecting the final clips.
I have a list of refined candidate highlights. Your task is to perform a deep scoring pass (Stage 3).

Your goals:
1. Improve titles to be highly clickable.
2. Identify the absolute strongest opening hook sentence from within the clip to maximize retention in the first 3 seconds. Prioritize shocking statements, bold claims, questions, surprising statistics, emotional openings, or conflict. DO NOT paraphrase. Quote the exact transcript sentence exactly in the hook_sentence field.
3. Score each clip using the detailed criteria (0-10 for each). Your scoring should heavily penalize clips that lack COMPLETENESS, VIRALITY, RETENTION, or NARRATIVE FLOW.
4. Identify the stages of the story arc present in the clip (Setup, Conflict, Rising Action, Peak, Resolution) and populate `story_arc_stages`.

Criteria:
1. Hook Strength: How compelling is the opening 3 seconds?
2. Curiosity Gap: Does it make the viewer want to stick around?
3. Emotional Impact: Does it evoke strong emotion?
4. Storytelling: Is there a clear narrative arc?
5. Educational Value: Does the viewer learn something useful?
6. Entertainment Value: Is it highly entertaining?
7. Conflict: Is there a debate, disagreement, or struggle?
8. Surprise: Is there a twist or shocking fact?
9. Authority/Credibility: Does the speaker sound authoritative or credible?
10. Viewer Retention Prediction: Will it hold attention to the very end?
11. Shareability: Is the viewer likely to share this with friends?
12. Comment Potential: Does it encourage viewers to comment?
13. Story Completion Score: How complete is the mini-story arc (Setup to Resolution)?
14. Emotion Score: Does the transcript show excitement, laughter, surprise, frustration, or emphasis? (Deduct points for monotone, filler, long pauses).

Candidates:
{candidates_json}

Return the refined, merged, and improved list of clips.
Respond ONLY with valid JSON matching the schema.
"""


COHERENCE_OPTIMIZATION_PROMPT = """You are a senior YouTube Shorts editor finalizing a clip.
Your goal is to ensure the clip feels completely coherent. It MUST NOT cut off in the middle of an explanation, joke, story, or example.
I will provide the current clip's text, as well as the transcript text immediately before and after it.

If the clip lacks sufficient context or ends abruptly, identify the EXACT verbatim sentence from the 'Context Before' where the clip should start, or the EXACT verbatim sentence from the 'Context After' where the clip should end. 
If the current boundaries are fine, return the current start and end sentences.
Avoid unnecessary length increases. Only expand if critical context is missing.

Current Clip:
{clip_text}

Context Before (60s):
{context_before}

Context After (60s):
{context_after}

Respond ONLY with valid JSON matching the schema.
"""

class CoherenceResult(BaseModel):
    new_start_sentence: str = Field(description="The exact verbatim sentence from the transcript where the clip should start.")
    new_end_sentence: str = Field(description="The exact verbatim sentence from the transcript where the clip should end.")

FINAL_REVIEW_PROMPT = """You are a Senior YouTube Shorts Editor performing the final quality assurance review on a generated clip.
Your goal is to ensure the clip is absolutely perfect before it is exported. 
Evaluate the clip against these criteria:
1. Does it start awkwardly? (e.g., mid-sentence, missing context)
2. Does it end awkwardly? (e.g., cut off before the payoff)
3. Does it feel incomplete?
4. Does it contain excessively long pauses or meandering speech?
5. Does it contain redundant or repeated information?

Decide whether to:
- "ACCEPT": The clip is great.
- "REJECT": The clip violates the criteria severely and cannot be salvaged.
- "MODIFY": The clip is mostly good but needs its boundaries shifted.

If you choose "MODIFY", you MUST provide the exact verbatim `new_start_sentence` and `new_end_sentence` from the provided context, and provide a `new_score`.

Current Clip ({duration:.1f}s):
{clip_text}

Context Before:
{context_before}

Context After:
{context_after}

Respond ONLY with valid JSON matching the schema.
"""

class FinalReviewResult(BaseModel):
    decision: str = Field(description="Must be exactly 'ACCEPT', 'REJECT', or 'MODIFY'")
    reason: str = Field(description="A short 1-sentence explanation of the decision.")
    new_start_sentence: Optional[str] = Field(None, description="If MODIFY, the exact verbatim sentence to start on.")
    new_end_sentence: Optional[str] = Field(None, description="If MODIFY, the exact verbatim sentence to end on.")
    new_score: Optional[int] = Field(None, description="If MODIFY, the new adjusted viral score (0-100).")

METADATA_GENERATION_PROMPT = """You are a YouTube Shorts growth expert.
Generate the complete metadata package for this Short to maximize reach and engagement.

Clip Title Idea: {clip_title}
Clip Hook: {clip_hook}

Clip Transcript:
{clip_text}

Provide:
- An optimized Title
- Catchy Thumbnail Text (3-5 words max)
- A description with a dynamic CTA tailored to the content
- 5-8 highly relevant hashtags
- 10-15 keywords for backend tags
- The exact hook sentence (you can reuse the one provided or slightly improve it)
- A brief explanation of why this clip has viral potential

Respond ONLY with valid JSON matching the schema.
"""

SEMANTIC_DEDUPLICATION_PROMPT = """You are a YouTube Shorts Content Editor.
Your goal is to detect semantic duplicates in a list of candidate clips.
We prefer DIVERSITY over quantity. If two clips cover the same core concept (repeated stories, advice, jokes, or explanations), you must reject the weaker one.

I will provide a JSON list of candidate clips, including their text.
Review them and return the list. For any clip that is a semantic duplicate of an earlier clip in the list, set `rejection_reason` explaining exactly which clip it duplicates and why (e.g., "Repeats the same story about investing as Clip #3").

Candidates:
{candidates_json}

Respond ONLY with valid JSON matching the schema.
"""

CHUNK_SIZE_SECONDS = 1200
LONG_VIDEO_THRESHOLD = 1800
CHUNK_OVERLAP_SECONDS = 60


def detect_content_type(transcript: Transcript, provider: BaseLLMProvider) -> ContentInfo:
    """Detect the content type and density of a transcript sample."""
    sample = " ".join(s.text for s in transcript.segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        return provider.generate_structured(prompt, ContentInfo)
    except Exception as e:
        logger.warning(f"Failed to detect content type: {e}. Defaulting to other/medium.")
        return ContentInfo(content_type="other", density="medium")


import concurrent.futures
from tqdm import tqdm

def generate_candidates_stage1(transcript: Transcript, provider: BaseLLMProvider, content_info: ContentInfo) -> List[Highlight]:
    """Stage 1: Generate 25 candidate highlights."""
    logger.info("Stage 1: Generating candidates...")
    candidates = []
    
    if transcript.duration >= LONG_VIDEO_THRESHOLD:
        start = 0
        chunks = []
        while start < transcript.duration:
            end = min(start + CHUNK_SIZE_SECONDS, transcript.duration)
            chunk_segs = [
                s for s in transcript.segments
                if s.start >= start and s.end <= end + CHUNK_OVERLAP_SECONDS
            ]
            if chunk_segs:
                chunks.append(chunk_segs)
            start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
            
        def _process_chunk(chunk_segs):
            chunk_text = "\n".join(f"[{s.start:.1f}s] {s.text.strip()}" for s in chunk_segs)
            prompt = STAGE1_PROMPT.format(content_type=content_info.content_type, density=content_info.density) + f"\n\nTranscript:\n{chunk_text}"
            try:
                result = provider.generate_structured(prompt, HighlightList)
                return result.highlights
            except Exception as e:
                logger.warning(f"Failed to generate candidates for chunk: {e}")
                return []
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunks), desc="Generating Candidates"):
                candidates.extend(future.result())
    else:
        prompt = STAGE1_PROMPT.format(content_type=content_info.content_type, density=content_info.density) + f"\n\nTranscript:\n{transcript.text}"
        try:
            result = provider.generate_structured(prompt, HighlightList)
            candidates.extend(result.highlights)
        except Exception as e:
            logger.warning(f"Failed to generate candidates: {e}")
            
    logger.info(f"Stage 1 Complete: {len(candidates)} candidates generated.")
    return candidates


def review_candidates_stage2(candidates: List[Highlight], provider: BaseLLMProvider) -> List[Highlight]:
    """Stage 2: Review, eliminate weak clips, and merge overlaps."""
    if not candidates:
        return []
        
    logger.info("Stage 2: Running elimination and refinement pass...")
    
    refined_candidates = []
    batches = [candidates[i:i+15] for i in range(0, len(candidates), 15)]
    
    def _review_batch(batch):
        batch_dicts = [{"title": c.title, "hook_sentence": c.hook_sentence, "start_time": c.start_time, "end_time": c.end_time} for c in batch]
        prompt = STAGE2_ELIMINATION_PROMPT.format(candidates_json=json.dumps(batch_dicts, indent=2))
        try:
            result = provider.generate_structured(prompt, HighlightList)
            return result.highlights
        except Exception as e:
            logger.warning(f"Failed to review batch in Stage 2: {e}")
            return batch
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_review_batch, batch) for batch in batches]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(batches), desc="Reviewing Candidates"):
            batch_results = future.result()
            for c in batch_results:
                if c.rejection_reason:
                    logger.info(f"Stage 2 Elimination: '{c.title}' rejected because {c.rejection_reason}")
                refined_candidates.append(c)
            
    logger.info(f"Stage 2 Complete: {len(refined_candidates)} clips returned by review pass.")
    return refined_candidates


def score_and_select_stage3(candidates: List[Highlight], provider: BaseLLMProvider) -> List[Highlight]:
    """Stage 3: Deep scoring based on completeness, virality, retention, and narrative flow."""
    # Filter out already rejected clips before scoring
    survivors = [c for c in candidates if not c.rejection_reason]
    
    if not survivors:
        return []
        
    logger.info(f"Stage 3: Running deep scoring on {len(survivors)} surviving clips...")
    
    scored_candidates = []
    batches = [survivors[i:i+15] for i in range(0, len(survivors), 15)]
    
    def _score_batch(batch):
        batch_dicts = [{"title": c.title, "hook_sentence": c.hook_sentence, "start_time": c.start_time, "end_time": c.end_time} for c in batch]
        prompt = STAGE3_SCORING_PROMPT.format(candidates_json=json.dumps(batch_dicts, indent=2))
        try:
            result = provider.generate_structured(prompt, HighlightList)
            refined = []
            for c in result.highlights:
                if c.criteria:
                    c.score = c.criteria.total
                    logger.info(f"Stage 3 Scored: '{c.title}' received score {c.score}")
                refined.append(c)
            return refined
        except Exception as e:
            logger.warning(f"Failed to score batch in Stage 3: {e}")
            return batch
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_score_batch, batch) for batch in batches]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(batches), desc="Scoring Candidates"):
            scored_candidates.extend(future.result())
            
    logger.info(f"Stage 3 Complete: {len(scored_candidates)} clips deeply scored.")
    return scored_candidates


def remove_duplicates_semantic(candidates: List[Highlight], transcript: Transcript, provider: BaseLLMProvider) -> List[Highlight]:
    """Stage 3: Remove physical and semantic duplicates."""
    logger.info("Stage 3: Removing physical and semantic duplicates...")
    
    # 1. Physical (Mathematical) Deduplication
    sorted_cands = sorted(candidates, key=lambda x: x.start_time)
    kept_phys = []
    
    for h in sorted_cands:
        if h.rejection_reason:
            kept_phys.append(h)
            continue
            
        overlapping = False
        for k in kept_phys:
            if k.rejection_reason:
                continue
                
            latest_start = max(h.start_time, k.start_time)
            earliest_end = min(h.end_time, k.end_time)
            overlap = earliest_end - latest_start
            
            if overlap > 0 and overlap > 0.5 * (h.end_time - h.start_time):
                overlapping = True
                break
                
        if not overlapping:
            kept_phys.append(h)
            
    # 2. Semantic Deduplication
    # Filter out already rejected clips for the semantic pass to save tokens
    to_check = [c for c in kept_phys if not c.rejection_reason]
    
    if len(to_check) < 2:
        return kept_phys
        
    logger.info(f"Running semantic deduplication on {len(to_check)} unique boundary clips...")
    
    batch_dicts = []
    for i, c in enumerate(to_check):
        clip_segs = [seg for seg in transcript.segments if seg.end >= c.start_time and seg.start <= c.end_time]
        clip_text = " ".join(s.text.strip() for s in clip_segs)[:500] # Provide snippet to save tokens
        batch_dicts.append({
            "id": f"Clip #{i+1}",
            "title": c.title, 
            "hook_sentence": c.hook_sentence, 
            "clip_text": clip_text
        })
        
    prompt = SEMANTIC_DEDUPLICATION_PROMPT.format(candidates_json=json.dumps(batch_dicts, indent=2))
    
    try:
        result = provider.generate_structured(prompt, HighlightList)
        for i, c in enumerate(result.highlights):
            if c.rejection_reason:
                to_check[i].rejection_reason = c.rejection_reason
                logger.info(f"Semantic Duplicate Rejected: '{to_check[i].title}' because {c.rejection_reason}")
    except Exception as e:
        logger.warning(f"Failed semantic deduplication: {e}")
        
    logger.info(f"Deduplication Complete: {len([c for c in kept_phys if not c.rejection_reason])} candidates remain.")
    return kept_phys


def is_valid_start(seg_text: str, prev_seg_text: str) -> bool:
    """Check if the segment text is a valid natural starting point."""
    seg_text = seg_text.strip()
    prev_seg_text = prev_seg_text.strip()
    
    if not prev_seg_text:
        return True
        
    if prev_seg_text[-1] in ".!?])" or prev_seg_text.endswith("--") or prev_seg_text.endswith("..."):
        return True
        
    if seg_text.startswith("-") or seg_text.startswith(">>") or seg_text.startswith("[") or seg_text.startswith("("):
        return True
        
    return False


def is_valid_end(seg_text: str, next_seg_text: str) -> bool:
    """Check if the segment text is a valid natural ending point."""
    seg_text = seg_text.strip()
    next_seg_text = next_seg_text.strip()
    
    if not seg_text:
        return False
        
    if seg_text[-1] in ".!?])" or seg_text.endswith("--") or seg_text.endswith("..."):
        return True
        
    if next_seg_text.startswith("-") or next_seg_text.startswith(">>") or next_seg_text.startswith("[") or next_seg_text.startswith("("):
        return True
        
    return False


def expand_boundaries(candidates: List[Highlight], transcript: Transcript) -> List[Highlight]:
    """Stage 4: Expand clip boundaries naturally so clips don't begin or end abruptly."""
    logger.info("Stage 4: Expanding clip boundaries...")
    segments = transcript.segments
    
    for h in candidates:
        if h.rejection_reason:
            continue
            
        start_idx = None
        end_idx = None
        
        for i, seg in enumerate(segments):
            if start_idx is None and seg.end >= h.start_time:
                start_idx = i
            if seg.start <= h.end_time:
                end_idx = i
                
        if start_idx is None or end_idx is None:
            continue
            
        # Extend backwards to a natural start
        while start_idx > 0:
            if is_valid_start(segments[start_idx].text, segments[start_idx - 1].text):
                break
            # Hard limit so we don't expand indefinitely (e.g. 20 seconds max)
            if h.start_time - segments[start_idx - 1].start > 20:
                break
            start_idx -= 1
            
        # Extend forwards to a natural end
        while end_idx < len(segments) - 1:
            if is_valid_end(segments[end_idx].text, segments[end_idx + 1].text):
                break
            # Hard limit so we don't expand indefinitely
            if segments[end_idx + 1].end - h.end_time > 20:
                break
            end_idx += 1
                
        h.start_time = segments[start_idx].start
        h.end_time = segments[end_idx].end
            
    logger.info("Stage 4 Complete: Boundaries expanded to natural sentences/thoughts.")
    return candidates


def filter_and_rank(candidates: List[Highlight], num_clips: int) -> List[Highlight]:
    """Stage 5: Return only the highest scoring clips. Reject short clips and specific tropes."""
    logger.info("Stage 5: Filtering and ranking...")
    
    valid_clips = []
    for c in candidates:
        if c.rejection_reason:
            logger.info(f"Rejected '{c.title}': {c.rejection_reason}")
            continue
            
        duration = c.end_time - c.start_time
        if duration < 20 and c.score <= 90:
            logger.info(f"Rejected '{c.title}': Too short ({duration:.1f}s) and score not exceptionally high.")
            c.rejection_reason = "Too short"
            continue
            
        valid_clips.append(c)
        
    valid_clips.sort(key=lambda x: x.score, reverse=True)
    
    final_clips = valid_clips[:num_clips]
    logger.info(f"Stage 5 Complete: Top {len(final_clips)} clips selected.")
    return final_clips


def ensure_clip_coherence(candidates: List[Highlight], transcript: Transcript, provider: BaseLLMProvider) -> List[Highlight]:
    """Context Extension AI Pass to ensure the clip doesn't cut off jokes or stories."""
    if not candidates:
        return candidates
        
    logger.info(f"Running Coherence Optimization on {len(candidates)} final clips...")
    
    import re
    
    for c in candidates:
        clip_segs = []
        before_segs = []
        after_segs = []
        
        for seg in transcript.segments:
            if seg.end < c.start_time:
                if c.start_time - seg.start <= 60:
                    before_segs.append(seg)
            elif seg.start > c.end_time:
                if seg.end - c.end_time <= 60:
                    after_segs.append(seg)
            else:
                clip_segs.append(seg)
                
        clip_text = " ".join(s.text.strip() for s in clip_segs)
        context_before = " ".join(s.text.strip() for s in before_segs)
        context_after = " ".join(s.text.strip() for s in after_segs)
        
        prompt = COHERENCE_OPTIMIZATION_PROMPT.format(
            clip_text=clip_text,
            context_before=context_before,
            context_after=context_after
        )
        
        try:
            result = provider.generate_structured(prompt, CoherenceResult)
            
            if result.new_start_sentence:
                target_clean = re.sub(r'[^\w\s]', '', result.new_start_sentence.lower().strip())
                if len(target_clean) > 10:
                    search_segs = before_segs + clip_segs
                    for seg in search_segs:
                        seg_clean = re.sub(r'[^\w\s]', '', seg.text.lower().strip())
                        if len(seg_clean) > 5 and (seg_clean in target_clean or target_clean in seg_clean):
                            if c.start_time != seg.start:
                                logger.info(f"Coherence Optimized for '{c.title}': shifted start from {c.start_time:.1f} to {seg.start:.1f}")
                                c.start_time = seg.start
                            break
                            
            if result.new_end_sentence:
                target_clean = re.sub(r'[^\w\s]', '', result.new_end_sentence.lower().strip())
                if len(target_clean) > 10:
                    search_segs = clip_segs + after_segs
                    for seg in reversed(search_segs):
                        seg_clean = re.sub(r'[^\w\s]', '', seg.text.lower().strip())
                        if len(seg_clean) > 5 and (seg_clean in target_clean or target_clean in seg_clean):
                            if c.end_time != seg.end:
                                logger.info(f"Coherence Optimized for '{c.title}': shifted end from {c.end_time:.1f} to {seg.end:.1f}")
                                c.end_time = seg.end
                            break
                            
        except Exception as e:
            logger.warning(f"Failed coherence optimization for '{c.title}': {e}")
            
    return candidates


def optimize_clip_hooks(candidates: List[Highlight], transcript: Transcript) -> List[Highlight]:
    """Snap clip boundaries to the exact timestamp of the AI's chosen hook sentence."""
    logger.info("Optimizing clip hooks...")
    
    import re
    import difflib
    
    for c in candidates:
        if not c.hook_sentence:
            continue
            
        target_clean = re.sub(r'[^\w\s]', '', c.hook_sentence.lower().strip())
        if len(target_clean) < 10:
            continue
            
        best_start = c.start_time
        
        for seg in transcript.segments:
            # Only consider segments near the original start to avoid jumping too far
            if seg.start >= c.start_time - 15 and seg.start <= c.start_time + 30:
                seg_clean = re.sub(r'[^\w\s]', '', seg.text.lower().strip())
                
                if len(seg_clean) > 10 and (seg_clean in target_clean or target_clean in seg_clean):
                    best_start = seg.start
                    break
                    
                if len(seg_clean) > 10:
                    ratio = difflib.SequenceMatcher(None, seg_clean, target_clean).ratio()
                    if ratio > 0.8:
                        best_start = seg.start
                        break
                        
        if best_start != c.start_time:
            logger.info(f"Hook Optimized for '{c.title}': shifted start from {c.start_time:.1f} to {best_start:.1f} to match hook '{c.hook_sentence}'")
            c.start_time = best_start
            
    return candidates


def human_editing_pass(candidates: List[Highlight], transcript: Transcript, provider: BaseLLMProvider) -> List[Highlight]:
    """Simulate a Senior Editor's final QA review on each clip."""
    logger.info("Executing Final Human Editing Pass...")
    import re
    
    final_passed_clips = []
    
    for c in candidates:
        clip_segs = [s for s in transcript.segments if s.end >= c.start_time and s.start <= c.end_time]
        clip_text = " ".join(s.text.strip() for s in clip_segs)
        
        before_segs = [s for s in transcript.segments if s.end < c.start_time and s.end >= c.start_time - 60]
        context_before = " ".join(s.text.strip() for s in before_segs)
        
        after_segs = [s for s in transcript.segments if s.start > c.end_time and s.start <= c.end_time + 60]
        context_after = " ".join(s.text.strip() for s in after_segs)
        
        prompt = FINAL_REVIEW_PROMPT.format(
            duration=c.end_time - c.start_time,
            clip_text=clip_text,
            context_before=context_before,
            context_after=context_after
        )
        
        try:
            resp_str = provider.generate_json(prompt, schema=FinalReviewResult.model_json_schema())
            result = FinalReviewResult.model_validate_json(resp_str)
            
            logger.info(f"Final Review for '{c.title}': {result.decision} - {result.reason}")
            
            if result.decision == "REJECT":
                continue
            elif result.decision == "MODIFY":
                if result.new_score is not None:
                    c.viral_score = result.new_score
                    
                if result.new_start_sentence:
                    target_clean = re.sub(r'[^\w\s]', '', result.new_start_sentence.lower().strip())
                    if len(target_clean) > 10:
                        search_segs = before_segs + clip_segs
                        for seg in search_segs:
                            seg_clean = re.sub(r'[^\w\s]', '', seg.text.lower().strip())
                            if len(seg_clean) > 5 and (seg_clean in target_clean or target_clean in seg_clean):
                                logger.info(f"Final Review shifted start from {c.start_time:.1f} to {seg.start:.1f}")
                                c.start_time = seg.start
                                break
                                
                if result.new_end_sentence:
                    target_clean = re.sub(r'[^\w\s]', '', result.new_end_sentence.lower().strip())
                    if len(target_clean) > 10:
                        search_segs = clip_segs + after_segs
                        for seg in reversed(search_segs):
                            seg_clean = re.sub(r'[^\w\s]', '', seg.text.lower().strip())
                            if len(seg_clean) > 5 and (seg_clean in target_clean or target_clean in seg_clean):
                                logger.info(f"Final Review shifted end from {c.end_time:.1f} to {seg.end:.1f}")
                                c.end_time = seg.end
                                break
                                
            final_passed_clips.append(c)
            
        except Exception as e:
            logger.warning(f"Failed Final Review for '{c.title}': {e}. Passing clip by default.")
            final_passed_clips.append(c)
            
    # Guarantee at least one clip returns if QA was too aggressive
    if not final_passed_clips and candidates:
        logger.warning("Final Review rejected all clips! Rescuing the highest scored clip.")
        best_clip = max(candidates, key=lambda x: x.viral_score)
        final_passed_clips.append(best_clip)
        
    return final_passed_clips


def generate_clip_metadata(candidates: List[Highlight], transcript: Transcript, provider: BaseLLMProvider) -> List[Highlight]:
    """Generate structured metadata for each final clip."""
    if not candidates:
        return candidates
        
    logger.info(f"Generating metadata for {len(candidates)} final clips...")
    
    for c in candidates:
        clip_segs = [seg for seg in transcript.segments if seg.end >= c.start_time and seg.start <= c.end_time]
        clip_text = " ".join(s.text.strip() for s in clip_segs)
        
        prompt = METADATA_GENERATION_PROMPT.format(
            clip_title=c.title,
            clip_hook=c.hook_sentence,
            clip_text=clip_text
        )
        
        try:
            metadata_result = provider.generate_structured(prompt, ShortMetadata)
            c.metadata = metadata_result
            logger.info(f"Metadata generated for '{c.title}'")
        except Exception as e:
            logger.warning(f"Failed to generate metadata for '{c.title}': {e}")
            
    return candidates


def apply_professional_padding(candidates: List[Highlight], transcript: Transcript) -> List[Highlight]:
    """
    Refine clip boundaries to be professional cuts.
    Finds the exact first and last word timestamps and pads them.
    start = first_word.start - 0.15s
    end = last_word.end + 0.20s
    """
    if not candidates:
        return candidates
        
    for c in candidates:
        first_word_start = None
        last_word_end = None
        
        for seg in transcript.segments:
            # Check segment overlap
            if seg.end >= c.start_time and seg.start <= c.end_time:
                if seg.words:
                    for w in seg.words:
                        # Allow a little slop around the strict bounds
                        if w.start >= c.start_time - 0.5 and w.end <= c.end_time + 0.5:
                            if first_word_start is None or w.start < first_word_start:
                                first_word_start = w.start
                            if last_word_end is None or w.end > last_word_end:
                                last_word_end = w.end
        
        final_start = c.start_time
        final_end = c.end_time
        
        if first_word_start is not None:
            final_start = max(0.0, first_word_start - 0.15)
        if last_word_end is not None:
            final_end = min(transcript.duration, last_word_end + 0.20)
            
        if final_start != c.start_time or final_end != c.end_time:
            logger.info(f"Professional Padding applied to '{c.title}': "
                        f"{c.start_time:.2f}-{c.end_time:.2f} -> {final_start:.2f}-{final_end:.2f}")
            c.start_time = final_start
            c.end_time = final_end
            
    return candidates


def apply_scene_aware_boundaries(candidates: List[Highlight], scenes: List[Tuple[float, float]]) -> List[Highlight]:
    """Snap clip boundaries to scene cuts if they are uncomfortably close."""
    if not scenes:
        return candidates
        
    scene_radius = 1.5  # search radius in seconds
    
    for c in candidates:
        orig_start = c.start_time
        orig_end = c.end_time
        
        # Snap start time: Prefer starting clips AFTER scene changes.
        # So we look for a scene cut (which is a scene end/start) near c.start_time.
        # If there's a scene cut within radius, snap start_time to it.
        # Actually, `scenes` is a list of (start, end). The cuts are basically the `start` (or `end`) values.
        # Let's collect all cut timestamps.
        cuts = [s[0] for s in scenes]
        if scenes:
            cuts.append(scenes[-1][1])
            
        best_start_cut = None
        min_start_dist = scene_radius
        for cut in cuts:
            dist = abs(cut - c.start_time)
            if dist < min_start_dist:
                min_start_dist = dist
                best_start_cut = cut
                
        if best_start_cut is not None:
            logger.info(f"Scene-Aware snapping start for '{c.title}': {c.start_time:.2f} -> {best_start_cut:.2f}")
            c.start_time = best_start_cut
            
        # Snap end time: Prefer ending clips BEFORE major camera transitions.
        # If there is a scene cut near c.end_time, snap to it.
        best_end_cut = None
        min_end_dist = scene_radius
        for cut in cuts:
            dist = abs(cut - c.end_time)
            if dist < min_end_dist:
                min_end_dist = dist
                best_end_cut = cut
                
        if best_end_cut is not None:
            # We want to end right before the cut
            logger.info(f"Scene-Aware snapping end for '{c.title}': {c.end_time:.2f} -> {best_end_cut:.2f}")
            c.end_time = best_end_cut
            
    return candidates


def get_highlights(transcript: Transcript, provider: BaseLLMProvider, scenes: List[Tuple[float, float]] = None, num_clips: int = 3) -> List[Highlight]:
    """Execute the full 3-stage AI ranking pipeline."""
    content_info = detect_content_type(transcript, provider)
    logger.info(f"Detected content: {content_info.content_type} (Density: {content_info.density})")
    
    # Stage 1: Generation
    candidates = generate_candidates_stage1(transcript, provider, content_info)
    
    # Stage 2: Elimination
    candidates = review_candidates_stage2(candidates, provider)
    
    # Deduplication (Physical & Semantic) and Boundary Expansion
    candidates = remove_duplicates_semantic(candidates, transcript, provider)
    candidates = expand_boundaries(candidates, transcript)
    
    # Stage 3: Scoring & Selection
    scored_candidates = score_and_select_stage3(candidates, provider)
    
    # Final Filtering & Ranking
    final_clips = filter_and_rank(scored_candidates, num_clips)
    
    # Coherence Optimization
    final_clips = ensure_clip_coherence(final_clips, transcript, provider)
    
    # Hook Optimization
    final_clips = optimize_clip_hooks(final_clips, transcript)
    
    # Final Human Editing Pass (QA)
    final_clips = human_editing_pass(final_clips, transcript, provider)
    
    # Scene-Aware Editing
    if scenes:
        final_clips = apply_scene_aware_boundaries(final_clips, scenes)
    
    # Professional Padding
    final_clips = apply_professional_padding(final_clips, transcript)
    
    # Metadata Generation
    final_clips = generate_clip_metadata(final_clips, transcript, provider)
    
    return final_clips
