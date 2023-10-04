

#docker run --rm --volume="$PWD:/srv/jekyll:Z" --publish [::1]:4000:4000 jekyll/jekyll jekyll serve

FROM jekyll/jekyll as builder

ADD Gemfile Gemfile.lock ./

RUN chmod a+w Gemfile.lock
#ADD Gemfile ./
RUN bundle install
#RUN bundle add webrick


#ARG JEKYLL_BASEURL=''
#
#####################################
#
#FROM ruby:alpine as builder
#
#RUN apk add --no-cache make build-base
#RUN gem install bundler -v 2.2.31
#
#WORKDIR /jekyll
#ADD Gemfile Gemfile.lock ./
#RUN bundle install
#
#ADD . .
#ARG JEKYLL_BASEURL
#RUN bundle exec jekyll build --baseurl $JEKYLL_BASEURL
#
#####################################
#
#FROM nginx:alpine
#
#ARG JEKYLL_BASEURL
#COPY --from=builder /jekyll/_site /usr/share/nginx/html/$JEKYLL_BASEURL
#COPY nginx.conf /etc/nginx/nginx.conf
#
#EXPOSE 80
