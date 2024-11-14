from sec_api import QueryApi, XbrlApi
import re

queryApi = QueryApi(api_key="YOUR_API_KEY")
xbrlApi = XbrlApi(api_key="YOUR_API_KEY")

query = {
    "query": "formType:\"10-K\" AND dataFiles.description:\"EXTRACTED XBRL INSTANCE DOCUMENT\"",
    "from": "0",
    "size": "10",  # just a sample. We need to adjust this based on which tickers we need
    "sort": [{"filedAt": {"order": "desc"}}]
}

filings = queryApi.get_filings(query)


def extract_financial_metrics(xbrl_url):
    xbrl_json = xbrlApi.xbrl_to_json(xbrl_url=xbrl_url)
    income_statement = xbrl_json.get("StatementsOfIncome", {})
    balance_sheet = xbrl_json.get("BalanceSheets", {})
    cash_flow_statement = xbrl_json.get("StatementsOfCashFlows", {})

    revenue_items = income_statement.get("Revenues", []) or income_statement.get(
        "RevenueFromContractWithCustomerExcludingAssessedTax", [])
    net_income_items = income_statement.get("NetIncomeLoss", [])

    revenue = revenue_items[0]["value"] if revenue_items else 0
    net_income = net_income_items[0]["value"] if net_income_items else 0

    return revenue, net_income

financial_data = []

for filing in filings["filings"]:
    for dataFile in filing.get("dataFiles", []):
        if dataFile["description"] == "EXTRACTED XBRL INSTANCE DOCUMENT":
            xbrl_url = dataFile["documentUrl"]
            revenue, net_income = extract_financial_metrics(xbrl_url)

            financial_data.append({
                "filedAt": filing["filedAt"],
                "formType": filing["formType"],
                "revenue": revenue,
                "net_income": net_income
            })
            break

performance_categories = []

for data in financial_data:
    net_income = data["net_income"]

    if net_income > 0:
        performance_category = "Positive"
    else:
        performance_category = "Negative"

    performance_categories.append({
        "filedAt": data["filedAt"],
        "formType": data["formType"],
        "revenue": data["revenue"],
        "net_income": net_income,
        "performanceCategory": performance_category
    })

print(performance_categories)

