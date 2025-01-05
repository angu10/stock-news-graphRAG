from typing import List, Dict, Any


def json_cleanup(result):
    chars_to_remove = ["```json", "```"]
    for char in chars_to_remove:
        result = result.replace(char, "", -1)
    return result


def get_relevant_charts(messages: List[Dict]) -> List[Dict]:
    """Get relevant charts from message history"""
    charts = []
    for message in messages:
        if message.get("charts"):
            charts.extend(message["charts"])
    return charts


def store_chart_in_message(message: Dict, chart: Dict):
    """Store chart in message for later retrieval"""
    if "charts" not in message:
        message["charts"] = []
    message["charts"].append(chart)
