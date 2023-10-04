# How to Run with Ruby on Winodws
## Ruby 설치
- [windows에서 Ruby 설치](https://rubyinstaller.org/downloads/)
(msys 설치 필요, ridk install에서는 그냥 Enter)

## 빌드
```bash
$ gem install bundler
$ cd <PROJECT_FOLDER>
$ bundle install
```

## Run
```bash
$ bundle exec jekyll serve
```

# How to Run with Docker
```bash
$ docker run --rm -v "${PWD}:/srv/jekyll:Z" -p 4000:4000 -dit jekyll/jekyll jekyll serve --force_polling
```

# Ref.
- [Original Repogitory](https://github.com/aksakalli/jekyll-doc-theme)