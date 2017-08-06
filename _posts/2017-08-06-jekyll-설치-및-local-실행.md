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
[http://localhost:4000][localhostURL]

[localhostURL]: https://localhost:4000
