import re
a = re.sub("[^0-9a-zA-Z ]",        # Anything except 0..9, a..z and A..Z
       "",                    # replaced with nothing
       "this is a test!!")    # in this string

print(a)