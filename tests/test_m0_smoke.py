from turbotext import FuzzyConfig, KeywordStore, Match, MatchPolicy


def test_import_public_api() -> None:
    assert KeywordStore is not None
    assert MatchPolicy is not None
    assert FuzzyConfig is not None
    assert Match is not None


def test_empty_store_extract() -> None:
    store = KeywordStore()
    assert store.extract("hello world") == []


def test_empty_store_replace() -> None:
    store = KeywordStore()
    text = "hello world"
    assert store.replace(text) == text


def test_add_keyword_returns_id() -> None:
    store = KeywordStore()
    kid = store.add_keyword("aspirin", canonical="Aspirin", category="DRUG")
    assert isinstance(kid, str) and kid


def test_all_policies_importable() -> None:
    for policy in MatchPolicy:
        store = KeywordStore(policy=policy)
        assert store.extract("") == []


def test_fuzzy_config_default() -> None:
    cfg = FuzzyConfig()
    assert cfg.max_edit_distance == 0


def test_fuzzy_config_custom() -> None:
    store = KeywordStore(fuzzy=FuzzyConfig(max_edit_distance=2))
    assert store.extract("hello") == []


def test_add_keywords_list() -> None:
    store = KeywordStore()
    ids = store.add_keywords(["aspirin", "ibuprofen", "tylenol"])
    assert len(ids) == 3
    assert all(isinstance(i, str) for i in ids)
    found = {m.canonical for m in store.extract("take aspirin and ibuprofen")}
    assert found == {"aspirin", "ibuprofen"}


def test_add_keywords_dict() -> None:
    store = KeywordStore()
    ids = store.add_keywords({"aspirin": "Aspirin", "ibuprofen": "Ibuprofen"})
    assert len(ids) == 2
    found = {m.canonical for m in store.extract("take aspirin and ibuprofen")}
    assert found == {"Aspirin", "Ibuprofen"}


def test_add_keywords_list_with_category_and_priority() -> None:
    store = KeywordStore()
    store.add_keywords(["aspirin", "ibuprofen"], category="DRUG", priority=5.0)
    matches = store.extract("take aspirin")
    assert matches[0].category == "DRUG"
    assert matches[0].priority == 5.0


def test_add_keywords_empty_list() -> None:
    store = KeywordStore()
    assert store.add_keywords([]) == []


def test_add_keywords_empty_dict() -> None:
    store = KeywordStore()
    assert store.add_keywords({}) == []


def test_add_keywords_dict_canonical_used_in_replace() -> None:
    store = KeywordStore()
    store.add_keywords({"aspirin": "Aspirin", "tylenol": "Acetaminophen"})
    assert store.replace("take aspirin or tylenol") == "take Aspirin or Acetaminophen"


def test_add_keywords_canonical_to_aliases() -> None:
    store = KeywordStore()
    keyword_dict = {
        "java": ["java_2e", "java programing"],
        "product management": ["PM", "product manager"],
    }
    ids = store.add_keywords(keyword_dict)
    assert len(ids) == 4  # 2 aliases each

    matches = store.extract("she studies java programing and PM")
    canonicals = {m.canonical for m in matches}
    assert canonicals == {"java", "product management"}


def test_add_keywords_aliases_resolve_to_same_canonical() -> None:
    store = KeywordStore()
    store.add_keywords({"aspirin": ["aspirin", "asprin", "aspirin tablet"]})
    for text in ["take aspirin", "take asprin", "take aspirin tablet"]:
        matches = store.extract(text)
        assert len(matches) == 1
        assert matches[0].canonical == "aspirin"


def test_add_keywords_aliases_with_category_priority() -> None:
    store = KeywordStore()
    store.add_keywords(
        {"java": ["java_2e", "java programing"]},
        category="LANG",
        priority=9.0,
    )
    matches = store.extract("studying java programing")
    assert matches[0].category == "LANG"
    assert matches[0].priority == 9.0
    assert matches[0].canonical == "java"


def test_add_keywords_mixed_dict_shapes() -> None:
    # dict values can be str or list in the same call.
    # str value  → {surface: canonical}  (key is the surface form)
    # list value → {canonical: [surfaces]} (key is the canonical)
    store = KeywordStore()
    store.add_keywords({
        "java": ["java_2e", "java programing"],   # canonical → [surfaces]
        "py": "python",                            # surface   → canonical
    })
    assert store.extract("learning java_2e")[0].canonical == "java"
    assert store.extract("learning py")[0].canonical == "python"
