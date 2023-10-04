from pcapng import FileScanner
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

with open('samplePackets.pcapng', 'rb') as fp:
	scanner = FileScanner(fp)
	count = 0
	for block in scanner:
		count += 1
	print(count)
