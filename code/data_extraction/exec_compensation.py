from sec_api import ExecCompApi
import pandas as pd

execCompApi = ExecCompApi(api_key="YOUR_API_KEY")

# Example
ticker = "MSFT"
compensation_data = execCompApi.get_data(ticker)

comp_df = pd.DataFrame(compensation_data)

comp_df = comp_df[comp_df["position"].str.lower().str.contains("chief executive officer", na=False)]

comp_df = comp_df.sort_values(by="year")

comp_df["comp_change"] = comp_df["total"].pct_change() * 100

comp_df["comp_increase"] = comp_df["comp_change"].apply(lambda x: 1 if x > 0 else 0 if x < 0 else None)

comp_df = comp_df.dropna(subset=["comp_increase"])

features = comp_df[["year", "salary", "bonus", "stockAwards", "optionAwards", "nonEquityIncentiveCompensation", "otherCompensation"]]
target = comp_df["comp_increase"]
