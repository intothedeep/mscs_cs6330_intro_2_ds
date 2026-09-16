# 이 책의 빌드. 엔진은 XeLaTeX (한글 조판).
.PHONY: all clean distclean watch

all:
	latexmk -xelatex main.tex

watch:          ## 파일이 바뀔 때마다 자동 재빌드
	latexmk -xelatex -pvc main.tex

clean:          ## 중간 파일만 삭제 (PDF 는 남김)
	latexmk -c

distclean:      ## PDF 포함 전부 삭제
	latexmk -C
