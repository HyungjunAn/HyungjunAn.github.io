# How to Run

## Windows
### Ruby Installer 설치
- [Recommanded Version](https://github.com/oneclick/rubyinstaller2/releases/download/RubyInstaller-3.2.2-1/rubyinstaller-devkit-3.2.2-1-x64.exe)
- [Official Page](https://rubyinstaller.org/downloads/)
- msys 설치 필요, ridk install에서는 msys2 선택 후 Enter

### Build
```bash
$ gem install bundler
$ cd <PROJECT_FOLDER>
$ bundle install
```

### Run
```bash
$ bundle exec jekyll serve
```

## Docker
```bash
$ docker run --rm -v "${PWD}:/srv/jekyll:Z" -p 4000:4000 -dit jekyll/jekyll jekyll serve --force_polling
```

# Ref.
- [Original Repogitory](https://github.com/aksakalli/jekyll-doc-theme)