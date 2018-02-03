import os
import sys

if __name__=="__main__":
	# f = raw_input("File:")
	f = "-A"
	m = raw_input("Commit message:")
	if m=="":
		m = "debug"
	os.system("git add "+f)
	os.system("git commit -m \""+m+"\"")
	os.system("git push")