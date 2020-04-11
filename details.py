from lxml import html

with open(r'./details.html', "r") as f:
    page = f.read()
tree = html.fromstring(page)

tables = tree.xpath('//table')

records = []

for row in tables[5].xpath('tr')[1:]:
    tds = row.xpath('td')
    if tds[0].text == 'Fine':
        return print(tds[1].text)

print(records)
