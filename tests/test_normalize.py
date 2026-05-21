from app.agent import QueryPlan, _normalize_plan


def test_normalize_plan_defaults() -> None:
    snapshot = {
        "collections": ["sales"],
        "primary_collection": "sales",
        "primary_field": "amount",
        "first_word": "amount",
    }
    plan = QueryPlan(action="count")
    plan = _normalize_plan(plan, snapshot)

    assert plan.collection == "sales"
    assert plan.limit == 20
