lst_latests:
	for l in  ~/bentoml/models/*/latest; do echo $$(cat $$l); done
rm_oldest:
	bentoml models list |grep -v "Tag" | grep -v "22:32:" | awk '{print $1}' | xargs -I {oldest} bentoml models delete {oldest} --yes
du_bento:
	du -sh ~/bentoml/models/*
serve:
	bentoml serve src.service:ModelProbaAdmission --port 3001