import re

from inferx.assistant_sim import CONTACTS, T_APPT_NEW, _split_pool, make_dataset, make_long_users, make_user
from inferx.locomo import sessions


def all_text(sample):
    return " ".join(message.text for _, _, group in sessions(sample) for message in group)


def test_same_seed_gives_same_data():
    assert make_user(1, "train") == make_user(1, "train")
    assert make_user(1, "train") != make_user(2, "train")


def test_runner_can_read_the_layout_and_dates_are_in_order():
    data = sessions(make_user(1, "val"))
    assert len(data) > 5
    numbers = [number for number, _, _ in data]
    assert numbers == sorted(numbers)
    assert re.fullmatch(r"8:00 pm on \d+ \w+, 2025", data[0][1])


def test_every_message_has_a_label_and_evidence_exists():
    sample = make_user(3, "val")
    ids = {message.id for _, _, group in sessions(sample) for message in group}
    assert ids == set(sample["labels"])
    for item in sample["qa"]:
        assert set(item["evidence"]) <= ids
        assert bool(item["evidence"]) == (item["category"] != 5)


def test_updates_point_at_an_earlier_message_about_the_same_thing():
    sample = make_user(4, "train")
    order = [message.id for _, _, group in sessions(sample) for message in group]
    updates = [(key, label) for key, label in sample["labels"].items() if label["op"] == "update"]
    assert updates
    for key, label in updates:
        assert order.index(label["replaces"]) < order.index(key)
        assert sample["labels"][label["replaces"]]["entity"] == label["entity"]


def test_current_question_answer_comes_from_the_latest_change():
    sample = make_user(5, "train")
    messages = {message.id: message.text for _, _, group in sessions(sample) for message in group}
    stale = [item for item in sample["qa"] if item["stale_answer"] and "address" in item["question"]]
    assert stale
    for item in stale:
        text = messages[item["evidence"][0]].lower()
        assert item["answer"].lower() in text
        assert item["stale_answer"].lower() not in text


def test_noise_is_never_stored_and_codes_are_marked_sensitive():
    sample = make_user(6, "train")
    for label in sample["labels"].values():
        if label["source"] in ("promotions", "system"):
            assert not label["store"]
        if label["sensitive"]:
            assert label["op"] == "ignore"


def test_test_split_holds_out_names_and_wording():
    held_out = set(_split_pool(CONTACTS, "test"))
    assert held_out and not held_out & set(_split_pool(CONTACTS, "train"))
    assert not set(_split_pool(T_APPT_NEW, "test")) & set(_split_pool(T_APPT_NEW, "train"))
    data = make_dataset(train=5, val=1, test=3)
    train_text = " ".join(all_text(sample) for sample in data["train"])
    test_text = " ".join(all_text(sample) for sample in data["test"])
    speakers = {message.speaker for sample in data["test"] for _, _, group in sessions(sample) for message in group}
    train_speakers = {message.speaker for sample in data["train"] for _, _, group in sessions(sample) for message in group}
    assert not (speakers & set(CONTACTS)) & train_speakers
    assert train_text != test_text


def test_test_split_still_has_unanswerable_questions():
    sample = make_user(3000, "test")
    assert sum(item["category"] == 5 for item in sample["qa"]) == 5
    text = all_text(sample)
    for item in sample["qa"]:
        if item["category"] == 5 and "pay for the" in item["question"]:
            assert item["question"].split("pay for the ")[1].rstrip("?") not in text


def test_long_users_are_much_longer_and_stay_consistent():
    short, long = make_user(2000, "val"), make_long_users(count=1)[0]
    assert len(long["labels"]) > 4 * len(short["labels"])
    data = sessions(long)
    order = [message.id for _, _, group in data for message in group]
    assert set(order) == set(long["labels"])
    for key, label in long["labels"].items():
        if label["replaces"]:
            assert order.index(label["replaces"]) < order.index(key)
    for item in long["qa"]:
        assert set(item["evidence"]) <= set(order)
