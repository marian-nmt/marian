// -*- mode: groovy; indent-tabs-mode: nil; tab-width: 2 -*-
//
// This file is needed for continuous integration testing with Jenkins at
// http://vali.inf.ed.ac.uk/jenkins (log in with your github credentials)
// For further explanation of how this works, see
// https://jenkins.io/doc/book/pipeline/jenkinsfile/
//
// Note for Emacs users: use groovy mode for syntax highlighting;
// See the Readme.md at this repo for details:
// https://github.com/Groovy-Emacs-Modes/groovy-emacs-modes

pipeline {
  agent any
  
  stages {
    stage('Build') {
      steps {
        echo 'Building Marian ...'
        sh 'export PATH="/usr/local/cuda-8.0/bin:${PATH}"'
        echo "PATH is ${PATH}"
        echo "LD_LIBRARY_PATH is ${LD_LIBRARY_PATH}"
        echo "LIBRARY_PATH is ${LIBRARY_PATH}"
        sh 'mkdir -p build'
        sh 'cd build'
        sh 'pwd'
        // sh 'rm -rf  build && mkdir build && cd build && cmake .. && make -j'
      }
    }
  }
}
