from keras.models import load_model
from imutils import paths
from pathlib import Path
from lxml import html
from decimal import Decimal
from itertools import groupby
import numpy as np
import imutils
import pickle
import cv2
import requests
import re
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
from datetime import datetime

def resize_to_fit(image, width, height):
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image

with open('./labels.dat', "rb") as f:
    lb = pickle.load(f)

model = load_model("./captcha_model_98.3.hdf5")

def solve_captcha(img_array):
    image = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv3() else contours[0]
    char_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w / h > 1: # 1 - best value so far (hyperparam)
            half_width = int(w / 2)
            char_image_regions.append((x, y, half_width, h))
            char_image_regions.append((x + half_width, y, half_width, h))
        else:
            char_image_regions.append((x, y, w, h))
    char_image_regions = sorted(char_image_regions, key=lambda x: x[0])
    predictions = []
    for char_bounding_box in char_image_regions:
        x, y, w, h = char_bounding_box
        char_image = image[y:y + h, x:x + w]
        char_image = resize_to_fit(char_image, 20, 20)
        char_image = np.expand_dims(char_image, axis=2)
        char_image = np.expand_dims(char_image, axis=0)
        prediction = model.predict(char_image)
        char = lb.inverse_transform(prediction)[0]
        predictions.append(char)
    return "".join(predictions)

def get_fine(s, details_payload):
    details = s.post('https://wmq.etimspayments.com/pbw/ticketDetailAction.doh', data=details_payload)
    fine = ''
    tree = html.fromstring(details.content)
    tables = tree.xpath('//table')
    for row in tables[5].xpath('tr')[1:]:
        tds = row.xpath('td')
        if tds[0].text == 'Fine':
            return tds[1].text

def get_records(plateNumber):
    s = requests.Session()
    s.headers.update({
        'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:74.0) Gecko/20100101 Firefox/74.0',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://wmq.etimspayments.com/pbw/inputAction.doh',
        'upgrade-insecure-requests': '1'
    })
    r = s.get('https://wmq.etimspayments.com/pbw/include/sanfrancisco/input.jsp')
    c = s.get('https://wmq.etimspayments.com/pbw/CaptchaServlet.doh', stream=True)
    # print(c)
    solution = solve_captcha(np.asarray(bytearray(c.raw.read()), dtype=np.uint8))
    tree = html.fromstring(r.content)
    inputs = tree.xpath('//input')
    payload = {}
    for input in inputs:
        if input.get('size') == '10':
            payload[input.get('name')] = solution
        else:
            payload[input.get('name')] = input.get('value')
    payload['plateNumber'] = plateNumber
    payload['statePlate'] = 'CA'
    # print(r.text)
    # print(payload)
    results = s.post('https://wmq.etimspayments.com/pbw/inputAction.doh', data=payload)
    tree = html.fromstring(results.text)
    tables = tree.xpath('//table')
    records = []
    for row in tables[1].xpath('tr')[1:]:
        tds = row.xpath('td')
        rec = {
            'num' : tds[1].xpath('a')[0].text.strip(),
            'issue_date' : tds[2].text,
            'violation_code' : tds[3].text,
            'violation' : tds[4].text,
            'due' : tds[5].text
        }
        rec['fine'] = get_fine(s, {
            'ticketNumber': rec['num'],
            'clientcode': '19',
            'locale': 'en',
            'requestType': 'submit'
        })
        records.append(rec)
    total_fine = sum(map(lambda r: float(re.sub(r'[^\d.]', '', r['fine'])), records))
    total_due = sum(map(lambda r: float(re.sub(r'[^\d.]', '', r['due'])), records))
    groups = []
    for k, g in groupby(records, lambda r: r['violation']):
        groups.append({
            'type': k,
            'count': len(list(g))
        })
    groups = sorted(groups, key=lambda g: g['count'], reverse=True)
    return {
        'total_fine': total_fine,
        'total_due': total_due,
        'groups': groups,
        'records': records
    }



class handler(BaseHTTPRequestHandler):
  def do_GET(self):
    self.send_response(200)
    self.send_header('Content-type', 'application/json')
    self.end_headers()
    plateNumber = parse_qs(self.path[2:])['plateNumber'][0]
    self.wfile.write(bytes(json.dumps(get_records(plateNumber)), 'utf8')i
    return

# print(get_records('5BWH824'))


httpd = HTTPServer(('', 8003), handler)
httpd.serve_forever()
