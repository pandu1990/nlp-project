#!/usr/bin/env python3
import sys
import xlrd
from lxml import html
import requests

if len(sys.argv) != 3:
	print('2 args needed')
	exit()

start_row = int(sys.argv[1])
end_row = int(sys.argv[2])

tid_index = 0
title_index = 1
url_index = 2
sep = '\t'

wb = xlrd.open_workbook('nlp_dataset.xlsx')
sheet = wb.sheet_by_index(0) 

f = open("output.tsv", "w")
for row in range(start_row, end_row+1):
	print('Getting data for row: ', row)

	data = [sheet.cell_value(row, tid_index), sheet.cell_value(row, title_index)]

	url = sheet.cell_value(row, url_index)
	r = requests.get(url)
	if r.status_code == 200:
		tree = html.fromstring(r.content)
		rating_component = tree.xpath('//*[@id="title-overview-widget"]/div[1]/div[2]/div/div[2]/div[2]/div[contains(@class,"subtext")]/text()')
		if len(rating_component) >= 1:
			rating = rating_component[0].strip()
		else:
			continue
		data.append(rating)
		if tree.xpath('//*[@id="titleStoryLine"]/span[2]/a[2]/text()')[0] == 'Plot Synopsis':
			r = requests.get(url + 'plotsummary')
			tree = html.fromstring(r.content)
			plot_array = tree.xpath('//*[@id="plot-synopsis-content"]/li[1]/text()')
			plot = ' '.join([x.strip() for x in plot_array])
			data.append(plot.replace('\n', ' '))
	f.write(sep.join(data))
	f.write('\n')