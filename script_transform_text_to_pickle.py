import batchgenerator as bg
import os
import cPickle

#path_to_files = '/media/tassadar/Data/text/sales_books/txt/'
path_to_files = '/media/tassadar/Data/text/'

chars = []

text = ''

linecounter = 0
charcounter = 0
filecounter = 0

for filename in [os.listdir(path_to_files)[0]]:
    print 'Processing file #', filecounter, ': ', filename
    filecounter += 1
    f = open(path_to_files+filename, 'r')
    print 'Converting text...'
    text_file = f.read().lower()
    text_utf = unicode(text_file, encoding='utf-8')

    print 'Checking chars...'

    for c in text_utf:
        if c not in chars:
            chars.append(c)
            print 'Added char ', c
            charcounter += 1

    text += text_utf

cPickle.dump([chars, text], open('chars_text_declaration.p', 'w'))

print len(chars)
print chars

print text[0:10001]
