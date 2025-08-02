import re

def extract_score_percentage(text: str) -> float:
    """
    Extracts the score from a string formatted like 'Score: XX/YY'
    and returns the percentage as a float.
    
    Returns None if no valid score found.
    """
    match = re.search(r'(\d{1,2}(?:\.\d+)?)\s*/\s*(\d{1,2}(?:\.\d+)?)', text)
    if match:
        achieved = float(match.group(1))
        maximum = float(match.group(2))
        if maximum > 0:
            return achieved / maximum
    return None

def parse_video_comparison_output(text: str, threshold: float = 0.5):
    sections = {
        "video_a_description": "",
        "video_b_description": "",
        "motion_comparison": "",
        "similarity_score": "",
        "suggestions": "",
        "is_above_threshold": "NO"
    }

    # Regex patterns for section headers
    patterns = {
        "video_a_description": r"### Video A Motion Description\s*(.*?)\s*---",
        "video_b_description": r"### Video B Motion Description\s*(.*?)\s*---",
        "motion_comparison": r"### Motion Comparison\s*(.*?)\s*---",
        "similarity_score": r"### Similarity Score.*?Score:\s*([0-9.]+)/1",
        "suggestions": r"### Suggestions for Improvement.*?\s*(.*)",
    }

    # Extract sections
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if key == "similarity_score":
                score = float(match.group(1))
                sections["similarity_score"] = f"{score:.2f}"
                sections["is_above_threshold"] = "YES" if score >= threshold else "NO"
            else:
                sections[key] = match.group(1).strip()

    return sections



def convert_escaped_newlines(text: str) -> str:
    return text.replace("\\n", "\n")