from src.data.collector import is_relevant_title


def test_collector_skips_obvious_irrelevant_results():
    assert not is_relevant_title("HIStory: Past, Present and Future, Book I", "history")
    assert not is_relevant_title("Science (journal)", "science")
    assert not is_relevant_title("Nature (journal)", "nature")


def test_collector_accepts_topic_and_relevant_subtopics():
    assert is_relevant_title("Science", "science")
    assert is_relevant_title("Field (mathematics)", "mathematics")
    assert is_relevant_title("Information technology", "technology")
