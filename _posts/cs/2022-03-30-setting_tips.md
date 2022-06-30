---
layout: post
title: 환경 세팅 꿀팁
author: An Hyungjun
categories: []
tags: [summary]
---

PC 개발 환경 세팅 관련한 꿀팁을 모아놨다.

--------------------------------------------------------------------------------

# MS Store 없이 Windows 앱 설치

## WSL
- [공식 문서] [1]의 1 ~ 5 단계 수행 후 배포판을 직접 다운받아서 설치

[1]: https://docs.microsoft.com/ko-kr/windows/wsl/install-manual

## Windows Terminal
- [Blog](https://hackmd.io/@ss14/windows-terminal#)

--------------------------------------------------------------------------------

# VS Code ssh-key 등록

- Client에서 공개키, 암호키 쌍을 생성

```bash
# Power Shell
$ ssh-keygen -t rsa

# 생성되는 id_rsa, id_rsa.pub 파일 경로 확인
# 	ex. id_rsa: 		~/.ssh/id_rsa
# 	ex. id_rsa.pub: 	~/.ssh/id_rsa.pub
```

- 공개키를 Host에 등록
```bash
# Client
$ scp [id_rsa.pub path] [ID]@[HOST]:id_rsa.pub

# Host
$ cd ~
$ cat id_rsa.pub >> .ssh/authorized_keys
```

- VS Code ssh 설정에서 "IdentityFile" 옵션을 통해 비밀키 경로 지정
```bash
# 예시

Host ~~~
  HostName ~~
  User ~~
  ForwardAgent yes
  IdentityFile ~/.ssh/id_rsa
```

## 참고
- [Blog 1](https://otugi.tistory.com/344)
- [Blog 2](https://snwo.tistory.com/173)

