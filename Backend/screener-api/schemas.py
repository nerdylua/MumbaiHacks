from pydantic import BaseModel, Field

class RequestModel(BaseModel):
    query: str

class Screener_Query(BaseModel):
    screener_query: str = Field(description="Write your query for screener.in")

keywords = """
Search Query
You can customize the query below:

example:

.....................
Sales
OPM
Profit after tax
Market Capitalization
Sales latest quarter
Profit after tax latest quarter
YOY Quarterly sales growth
YOY Quarterly profit growth
Price to Earning
Dividend yield
Price to book value
Return on capital employed
Return on assets
Debt to equity
Return on equity
EPS
Debt
Promoter holding
Change in promoter holding
Earnings yield
Pledged percentage
Industry PE
Sales growth
Profit growth
Current price
Price to Sales
Price to Free Cash Flow
EVEBITDA
Enterprise Value
Current ratio
Interest Coverage Ratio
PEG Ratio
Return over 3months
Return over 6months
................................
+
-
/
*
>
<
AND
OR
...................

"""
