---
layout: post
title: 'Jekyll 설치 및 Local 실행'
comments: true
tags: [hyde]
---

## Ruby 설치
아래의 사이트에서 Ruby 2.3 최신버전을 받는다.
cd C:\RubyDevKit
ruby dk.rb init
ruby dk.rb install

## Ruby 설치
gem install jekyll

## Ruby Development kit 설치
  <!--then run code that depends on jQuery in the next script tag:-->

gem install bundler
bundle install

windows only
chcp 65001
bundle exec jekyll serve

~~~html
<script src=".../jquery.js"></script>
<script>
  $('#tabs').someJQueryFunction(); // works
</script>
~~~

I'd consider this an anti-pattern for the reason mentioned above,
but it remains common and has the advantage of being easy to understand.

However, things break when Hydejack dynamically inserts new content into the page.
It works fine for standard markdown content like `p` tags,
but when inserting `script` tags the browser will execute them immediately and in parallel,
because in most cases this is what you'd want.
However, this means that `$('#tabs').someJQueryFunction();` will run while the HTTP request for jQuery is still
in progress --- and we get an error that `$` isn't defined, or similar.

From this description the solution should be obvious: Insert the `script` tags one-by-one,
to simulate how they would get executed if it was a fresh page request.
In fact this is how Hydejack is now handling things (and thanks to rxjs' `concatMap` it was easy to implement),
but unfortunately this is not a magic solution that can fix all problems:

* Some scripts may throw when running on the same page twice
* Some scripts rely on the document's `load` event, which has fired long before the script was inserted
* unkown-unkowns

But what will "magically" solve all third party script problems, is disabling dynamic page loading altogether,
for which there's now an option.
To make this a slightly less bitter pill to swallow,
there's now a CSS-only "intro" animation that looks similar to the dynamic one.
Maybe you won't even notice the difference.

## Patch Notes
### Minor
* Support embedding `script` tags in markdown content
* Add `disable_push_state` option to `_config.yml`
* Add `disable_drawer` option to `_config.yml`
* Rename syntax highlighting file to `syntax.scss`
* Added [chapter on third party scripts][scripts] to documentation

### Design
* Add subtle intro animation
* Rename "Check out X for more" to "See X for more" on welcome\* page
* Replace "»" with "→" in "read more"-type of links

### Fixes
* Fix default color in gem-based theme

[scripts]: https://qwtel.com/hydejack/docs/scripts/
