find . -type f -not -path "*/.git/*" -not -name ".*" -exec file --mime {} \; \     130 ↵
| grep 'text/' | cut -d: -f1 | while read f; do echo -e "\n--- $f ---"; cat "$f"; done \
| pbcopy