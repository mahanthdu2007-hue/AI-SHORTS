import tempfile
import os
import subprocess

fd, ass_path = tempfile.mkstemp(suffix='.ass')
with os.fdopen(fd, 'w', encoding='utf-8') as f:
    f.write('[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 1\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,90,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,350,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,Hello\n')

# Create a dummy video
subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1920x1080:d=1', 'dummy.mp4'], check=True, capture_output=True)

print("Original ASS path:", ass_path)

def escape_1(p):
    return p.replace('\\', '/').replace(':', '\\:')

def escape_2(p):
    return f"'{p.replace('\\', '/').replace(':', '\\:')}'"

def escape_3(p):
    return f"filename='{p.replace('\\', '/').replace(':', '\\\\:')}'"

def escape_4(p):
    return p.replace('\\', '\\\\').replace(':', '\\\\:')

def escape_5(p):
    # Ffmpeg escapes for Windows:
    # forward slashes, and double escape colon because it passes through filter parser
    return p.replace('\\', '/').replace(':', '\\\\\\\\:')

def escape_6(p):
    # ffmpeg standard quoting
    return f"{p.replace('\\', '/').replace(':', '\\\\:')}"

for i, escape_method in enumerate([escape_1, escape_2, escape_3, escape_4, escape_5, escape_6], 1):
    esc_path = escape_method(ass_path)
    print(f'Trying {i}: ass={esc_path}')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', 'dummy.mp4', '-vf', f'ass={esc_path}', 'out.mp4']
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print('SUCCESS')
    except subprocess.CalledProcessError as e:
        print('FAILED:', e.stderr.strip())

os.remove(ass_path)
if os.path.exists('dummy.mp4'): os.remove('dummy.mp4')
if os.path.exists('out.mp4'): os.remove('out.mp4')
