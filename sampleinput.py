from pipeline.artic_engine import ArticFeedbackSystem

engine = ArticFeedbackSystem()

my_speech = {
    "id": "my_test",
    "duration": 8.0,
    "text_verbatim": "So um I was like um thinking like about kasi the results",
    "words": [
        {"word": "So",       "duration": 0.25, "start_time": 0.0,  "end_time": 0.25, "pause_before": 0.05, "pause_after": 0.05},
        {"word": "um",       "duration": 0.45, "start_time": 0.35, "end_time": 0.80, "pause_before": 0.35, "pause_after": 0.105},
        {"word": "I",        "duration": 0.15, "start_time": 1.10, "end_time": 1.25, "pause_before": 0.05, "pause_after": 0.03},
        {"word": "was",      "duration": 0.20, "start_time": 1.28, "end_time": 1.48, "pause_before": 0.03, "pause_after": 0.05},
        {"word": "like",     "duration": 0.28, "start_time": 1.53, "end_time": 1.81, "pause_before": 0.25, "pause_after": 0.15},
        {"word": "um",       "duration": 0.50, "start_time": 1.96, "end_time": 2.46, "pause_before": 0.40, "pause_after": 0.35},
        {"word": "thinking", "duration": 0.35, "start_time": 2.81, "end_time": 3.16, "pause_before": 0.05, "pause_after": 0.05},
        {"word": "like",     "duration": 0.25, "start_time": 3.21, "end_time": 3.46, "pause_before": 0.20, "pause_after": 0.12},
        {"word": "about",    "duration": 0.25, "start_time": 3.58, "end_time": 3.83, "pause_before": 0.05, "pause_after": 0.03},
        {"word": "kasi",     "duration": 0.30, "start_time": 3.86, "end_time": 4.16, "pause_before": 0.03, "pause_after": 0.05},
        {"word": "the",      "duration": 0.15, "start_time": 4.91, "end_time": 5.06, "pause_before": 0.05, "pause_after": 0.03},
        {"word": "results",  "duration": 0.30, "start_time": 5.09, "end_time": 5.39, "pause_before": 0.03, "pause_after": 0.0},
    ]
}

report = engine.analyze_speech_record(my_speech)
engine.save_and_display_report(report, student_name="My Custom Test")
