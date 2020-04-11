from lxml import html

with open(r'./page.html', "r") as f:
    page = f.read()
tree = html.fromstring(page)

tables = tree.xpath('//table')

records = []

for row in tables[1].xpath('tr')[1:]:
    tds = row.xpath('td')
    rec = {
        'num' : tds[1].xpath('a')[0].text.strip(),
        'issue_date' : tds[2],
        'violation_code' : tds[3],
        'violation' : tds[4],
        'due' : tds[5]
    }
    records.append(rec)

print(records)
