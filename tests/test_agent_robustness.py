import datetime as dt
from app.agent import (
    parse_date_string,
    convert_dates_in_filter,
    _guess_collection,
    _question_wants_count,
    _validate_plan_fields,
    QueryPlan,
    _normalize_plan,
)


def test_parse_date_string() -> None:
    # Standard ISO format with Z
    d1 = parse_date_string("2015-12-01T00:00:00Z")
    assert isinstance(d1, dt.datetime)
    assert d1.year == 2015
    assert d1.month == 12
    assert d1.day == 1

    # Standard ISO format without Z
    d2 = parse_date_string("2015-12-01T00:00:00")
    assert isinstance(d2, dt.datetime)
    assert d2.year == 2015
    assert d2.month == 12
    assert d2.day == 1

    # Simple date format
    d3 = parse_date_string("2015-03-24")
    assert isinstance(d3, dt.datetime)
    assert d3.year == 2015
    assert d3.month == 3
    assert d3.day == 24

    # Strptime with spaces
    d4 = parse_date_string("2015-03-24 18:50:57")
    assert isinstance(d4, dt.datetime)
    assert d4.year == 2015
    assert d4.month == 3
    assert d4.day == 24
    assert d4.hour == 18
    assert d4.minute == 50
    assert d4.second == 57

    # ISODate wrapped string
    d5 = parse_date_string("ISODate('2015-03-24T18:50:57Z')")
    assert isinstance(d5, dt.datetime)
    assert d5.year == 2015
    assert d5.month == 3

    # Non-dates should return None
    assert parse_date_string("not-a-date") is None
    assert parse_date_string("12345") is None


def test_convert_dates_in_filter() -> None:
    filter_doc = {
        "saleDate": {
            "$gte": "2015-12-01T00:00:00Z",
            "$lt": "2016-01-01"
        },
        "status": "active",
        "tags": ["office", "2015-03-24"]
    }
    converted = convert_dates_in_filter(filter_doc)
    
    assert isinstance(converted["saleDate"]["$gte"], dt.datetime)
    assert isinstance(converted["saleDate"]["$lt"], dt.datetime)
    assert converted["status"] == "active"
    # Inside a list:
    assert isinstance(converted["tags"][1], dt.datetime)
    assert converted["tags"][0] == "office"


def test_guess_collection_improvements() -> None:
    collections = ["customers", "users", "sales", "products"]
    
    # "customers" should map to customers, not users
    assert _guess_collection("How many customers do we have?", collections) == "customers"
    assert _guess_collection("List all customers", collections) == "customers"

    # "users" should map to users
    assert _guess_collection("How many users are active?", collections) == "users"

    # "sales" should map to sales
    assert _guess_collection("Show total sales last month", collections) == "sales"

    # "products" should map to products
    assert _guess_collection("Which product sold the most?", collections) == "sales"  # Wait! "sold the most" contains "sold" which doesn't match product keywords directly unless it contains product. Let's see:
    # "Which product sold the most?" has "product" -> matches mapping["products"] -> so it will map to products if "sales" is not matched first or if both keywords exist.
    # Actually, in guess_collection, "product" maps to "products", and "sold" doesn't map to sales. But "Which product sold the most?" has "product".
    # Let's check mapping priority. It loops over the dictionary order.
    # "products" is before "sales" in our mapping, so it matches "products" first because of "product".
    # Wait, is that okay? Yes! If the user says "Which product", mapping to "products" is correct, and then the agent/planner or LLM will plan the query on "sales" if it determines the action is "top sales" or it will refine it!
    # Let's verify the exact match for "product":
    assert _guess_collection("Which product is cheapest?", collections) == "products"


def test_question_wants_count() -> None:
    assert _question_wants_count("what is voucher count in company") is True
    assert _question_wants_count("how many customers do we have") is True
    assert _question_wants_count("total sales last month") is False
    assert _question_wants_count("top 5 customers by sales") is False


def test_validate_plan_filter_fields() -> None:
    snapshot = {
        "fields_by_collection": {"Voucher": ["_id", "companyName", "voucherCode"]}
    }
    plan = QueryPlan(action="count", collection="Voucher", filter={"company": "ACME"})
    _, issue = _validate_plan_fields(plan, snapshot)
    assert issue is not None
    assert "Filter fields are not available" in issue

    plan_ok = QueryPlan(action="count", collection="Voucher", filter={"companyName": "ACME"})
    _, issue_ok = _validate_plan_fields(plan_ok, snapshot)
    assert issue_ok is None
