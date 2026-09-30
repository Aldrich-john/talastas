"""
Talastas: Rule-Based Presentation Feedback Engine
Generates actionable, presentation-oriented coaching suggestions based on:
- Disfluency density (Disfluencies Per Minute - DPM)
- Breakdown of specific filler types (um, uh, like)
- Speaking rate (Words Per Minute - WPM)
"""

def generate_presentation_feedback(
    total_words: int,
    total_duration_sec: float,
    detected_disfluencies: list
) -> dict:
    """
    Evaluates speech metrics and produces rule-based suggestions.
    """
    duration_min = max(total_duration_sec / 60.0, 0.05)
    wpm = round(total_words / duration_min, 1)
    
    # Counts by category
    counts = {"um": 0, "uh": 0, "like_filler": 0}
    for item in detected_disfluencies:
        ftype = item.get("type")
        if ftype in counts:
            counts[ftype] += 1

    total_fillers = len(detected_disfluencies)
    dpm = round(total_fillers / duration_min, 1)

    suggestions = []
    strengths = []

    # 1. Pacing Evaluation
    if wpm < 110:
        pacing_status = "Slow Pacing"
        suggestions.append({
            "category": "Pacing & Delivery",
            "observation": f"Speaking rate is {wpm} WPM (below recommended 120-150 WPM).",
            "recommendation": "Try to maintain momentum between key points. Extended pauses might make the audience feel the presentation is dragging."
        })
    elif 110 <= wpm <= 155:
        pacing_status = "Optimal Pacing"
        strengths.append(f"Great speaking pace of {wpm} WPM! This allows clear comprehension in both English and Tagalog segments.")
    else:
        pacing_status = "Fast / Rushed"
        suggestions.append({
            "category": "Pacing & Delivery",
            "observation": f"Speaking rate is {wpm} WPM (above recommended 150 WPM).",
            "recommendation": "Slow down slightly. Rushing often triggers anxiety and unintended filler words when switching between Tagalog and English concepts."
        })

    # 2. Disfluency Density (DPM) Evaluation
    if dpm <= 2.0:
        fluency_grade = "High Clarity (Professional)"
        strengths.append(f"Excellent fluency with only {dpm} fillers per minute. Your speech is talastasulate and concise.")
    elif 2.0 < dpm <= 5.0:
        fluency_grade = "Moderate Fluency"
        suggestions.append({
            "category": "Filler Reduction",
            "observation": f"Observed {dpm} disfluencies per minute ({total_fillers} total).",
            "recommendation": "Practice comfortable silence. When transitioning between ideas, pause silently for 1 second instead of voicing a filler."
        })
    else:
        fluency_grade = "Needs Improvement"
        suggestions.append({
            "category": "Filler Reduction",
            "observation": f"High disfluency frequency: {dpm} fillers per minute ({total_fillers} total).",
            "recommendation": "Vocal fillers are noticeably interrupting your delivery flow. Rehearse with bullet points rather than reciting verbatim to reduce cognitive hesitation."
        })

    # 3. Targeted Feedback for Specific Fillers
    if (counts["um"] + counts["uh"]) > 2:
        suggestions.append({
            "category": "Hesitation Markers (um / uh)",
            "observation": f"Detected {counts['um']} 'um' and {counts['uh']} 'uh' occurrences.",
            "recommendation": "These hesitation sounds occur when thinking ahead. In Taglish presentations, take a breath at punctuation marks rather than saying 'um'/'uh' before Tagalog connector words like 'kasi' or 'kaya'."
        })

    if counts["like_filler"] > 1:
        suggestions.append({
            "category": "Colloquial Crutch ('like')",
            "observation": f"Detected {counts['like_filler']} instances of 'like' used as an informal filler crutch.",
            "recommendation": "In professional or academic presentations, replace filler 'like' with formal transitions such as 'for example', 'specifically', or Tagalog markers like 'halimbawa' or 'partikular'."
        })

    return {
        "summary": {
            "total_words": total_words,
            "duration_seconds": round(total_duration_sec, 2),
            "speaking_rate_wpm": wpm,
            "pacing_status": pacing_status,
            "total_disfluencies": total_fillers,
            "disfluencies_per_minute": dpm,
            "fluency_grade": fluency_grade,
            "counts_by_type": counts
        },
        "strengths": strengths,
        "recommendations": suggestions
    }
