#import csv
#with open('eggs.csv', 'wb') as csvfile:
#    spamwriter = csv.writer(csvfile, delimiter=' ',
#                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
#    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
#    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
import csv
data = [
  [' 101/01/31', '5,486,734,180', '162,361,181,834', '1,283,951', '7,517.08', '109.67'],
  [' 101/01/13', '3,497,838,901', '99,286,437,370', '819,762', '7,181.54', '-5.04'],
]
f = open("stock.csv","w")
w = csv.writer(f)
w.writerows(data)
f.close()