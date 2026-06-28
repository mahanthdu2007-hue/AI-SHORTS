import tempfile
import os
import subprocess

fd, ass_path = tempfile.mkstemp(prefix='test , name ', suffix='.ass')
with os.fdopen(fd, 'w', encoding='utf-8') as f:
    f.write('[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 1\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,90,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,350,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,Hello\n')

# Create a dummy video
subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=192x108:d=1', 'dummy.mp4'], check=True, capture_output=True)

def escape_test(p):
    # Convert to absolute path with forward slashes
    p = os.path.abspath(p).replace('\\', '/')
    # Escape colon (important for C:/...)
    p = p.replace(':', r'\:')
    # Escape single quote
    p = p.replace("'", r"\'")
    # Wrap in single quotes to protect spaces and commas
    return f"'{p}'"

esc_path = escape_test(ass_path)
print(f'Trying: ass={esc_path}')
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', 'dummy.mp4', '-vf', f'ass={esc_path}', 'out.mp4']
try:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print('SUCCESS')
except subprocess.CalledProcessError as e:
    print('FAILED:', e.stderr.strip())

os.remove(ass_path)
