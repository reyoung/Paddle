#!/bin/bash

function uninstall_paddle() {
	pip uninstall -y paddle py_paddle
	rm -rf /usr/local/bin/paddle /usr/local/opt/paddle
}

function build_install_paddle() {
	rm -rf /build
	mkdir -p /build
	cd /build
	cmake ${PADDLE_SOURCE_ROOT} -DWITH_GPU=${PADDLE_WITH_GPU}
	make install -j `nproc`
}


function ps_pid() {
	PID=$1
	ps -p ${PID} -o command,%cpu,%mem,time,vsz,rss
}