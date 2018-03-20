#! -*- coding:utf-8 -*-

f = open("develop.data", 'rb')
line_num = 0
last_q = ''
pos_cnt = 0
for l in f:
	line = l.decode('utf-8')
	line_item = line.split('\t')
	try:
		assert len(line_item) == 3
	except AssertionError:
		print("len of line != 3 at line %d. actual length is %d" % (line_num, len(line_item)))
	q = line_item[0]
	a = line_item[1]
	label = int(line_item[2])
	if q != last_q:
		if pos_cnt >= 2:
			print("more than 1 matched answer in question at line %d" % line_num)
		pos_cnt = 0
		last_q = q
	if label == 1:
		pos_cnt += 1
	line_num += 1


f.close()

