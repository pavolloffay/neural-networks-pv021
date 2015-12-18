ZIP_NAME=loffay_blahut_brandejs

all: zip

zip:
	mvn clean
	zip -r $(ZIP_NAME).zip -r * -x \
		'.idea/*' '*.iml' 'Makefile' 'target/*' '$(ZIP_NAME).zip' '.git/*' '.gitignore' 'travis.yml'

