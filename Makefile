.PHONY: lst_latests rm_oldest du_bento serve token predict test-api

PORT ?= 3001
SERVICE ?= src.service:ModelProbaAdmission
BASE_URL ?= http://127.0.0.1:$(PORT)


lst_latests:
	for l in  ~/bentoml/models/*/latest; do echo $$(cat $$l); done
rm_oldest:
	bentoml models list |grep -v "Tag" | grep -v "22:32:" | awk '{print $1}' | xargs -I {oldest} bentoml models delete {oldest} --yes
du_bento:
	du -sh ~/bentoml/models/*
serve:
	bentoml serve src.service:ModelProbaAdmission --port 3001

token:
	@curl -s -X POST "$(BASE_URL)/login" \
        -H "Content-Type: application/json" \
        -d '{"credentials":{"username":"user123","password":"password123"}}' | jq -r '.token'

predict:
	@token=$$(curl -s -X POST "$(BASE_URL)/login" \
        -H "Content-Type: application/json" \
        -d '{"credentials":{"username":"user123","password":"password123"}}' | jq -r '.token'); \
	curl -s -X POST "$(BASE_URL)/predict" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $$token" \
        -d '{"input_data": {"GRE Score": 340, "TOEFL Score": 200, "University Rating": 4, "SOP": 3, "LOR": 4, "CGPA": 4, "Research": 1}}'; \
	echo

test-api: predict