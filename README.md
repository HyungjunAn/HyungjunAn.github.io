# How to Use with Docker
```bash
# Update Gemfile.lock
$ docker run --rm -v "${PWD}:/srv/jekyll:Z" -it jekyll/jekyll bundle update

# Run Container
$ docker run --rm -v "${PWD}:/srv/jekyll:Z" -p 4000:4000 -dit jekyll/jekyll jekyll serve --force_polling --profile
```

# Ref.
- [Original Repo](https://aksakalli.github.io/jekyll-doc-theme/)