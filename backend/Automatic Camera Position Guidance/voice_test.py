import subprocess
import time


def speak(text):
    safe_text = text.replace("'", "''")

    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$speaker.Rate = 0; "
        "$speaker.Volume = 100; "
        f"$speaker.Speak('{safe_text}');"
    )

    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


print("Voice test started...")

speak("Camera guidance started")
time.sleep(1)

speak("Move closer")
time.sleep(1)

speak("Move farther")
time.sleep(1)

speak("Perfect position. Hold steady")

print("Voice test finished.")