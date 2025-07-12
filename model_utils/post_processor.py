import re

def extract_score_percentage(text: str) -> float:
    """
    Extracts the score from a string formatted like 'Score: XX/YY'
    and returns the percentage as a float.
    
    Returns None if no valid score found.
    """
    match = re.search(r'Score:\s*(\d+)\s*/\s*(\d+)', text)
    if match:
        achieved = int(match.group(1))
        maximum = int(match.group(2))
        if maximum > 0:
            return (achieved / maximum) * 100
    return None
