---
layout: post
title: 알쓸신잡
author: An Hyungjun
categories: []
tags: [summary]
---

문득문득 떠오르는 궁금증에 대한 해답을 정리하기 위한 포스팅이다.
주로 이걸 굳이 별도의 페이지로 정리를 해야 하나... 싶은 것들이 주를 이룬다.

--------------------------------------------------------------------------------

# TODO
- Process vs. Thread

--------------------------------------------------------------------------------

# VS.

## Framework vs. Library
- 표면적인 차이
	- Framework: 특정한 기능의 프로그램을 만들기 위해 상호 협력하는 클래스와 인터페이스의 집합
	- Library: 단순 활용이 가능한 도구들의 집합
- 제어의 역전(Inversion Of Control)에 의한 구분
	- Framework: 프레임워크가 프로그래머의 코드를 동작함
	- Library: 프로그래머의 코드가 라이브러리를 동작함
- 참고
	- [blog](https://mangkyu.tistory.com/4])

## UART vs. SPI vs. I2C
- 특징
	- UART: 비동기 시리얼 통신, 두 개의 핀으로 1:1 통신만 가능
	- SPI: 동기식 시리얼 통신, 1:N 통신 가능, 하드웨어 설계시 핀이 많이 필요
	- I2C: 동기식 시리얼 통신, N:N 통신 가능, 디바이스 당 핀을 2개(SCL, SDA)씩만 사용, 하드웨어적으로 단순, 소프트웨어 난이도가 상대적으로 높음(프로토콜)
- 참고
	- [blog](https://2innnnn0.tistory.com/11)
	- [blog](https://coder-in-war.tistory.com/entry/Network-02-I2C%EC%97%90-%EA%B4%80%ED%95%98%EC%97%AC)

--------------------------------------------------------------------------------

# Terminology

## GPIO(General-Purpose I/O)
- 개념: 마이크로 프로세서가 주변장치와 통신하기 위해 범용으로 사용되는 입출력 포트
- 참고
	- [wiki](https://ko.wikipedia.org/wiki/GPIO)
	- [blog](https://rakuraku.tistory.com/148)
	- [blog](https://junolefou.tistory.com/4)

--------------------------------------------------------------------------------

# Question

## 커널은 프로세스인가?
- 개념
	- 커널은 유저 프로세스와 별도의 주소 공간에서 실행되는 프로세스이다.
	- 커널은 멀티쓰레드로 구현되기도 하고 멀티프로세스로 구현되기도 한다.
- 참고
	- [KLDP](https://kldp.org/node/82997)