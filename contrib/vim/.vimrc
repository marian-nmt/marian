autocmd BufRead,BufNewFile *.cu set filetype=cpp
augroup cpp
    au!
    au BufRead,BufNewFile *.c,*.cpp,*.cu,*.h,*.hpp set tabstop=2
    au BufRead,BufNewFile *.c,*.cpp,*.cu,*.h,*.hpp set shiftwidth=2
    au BufRead,BufNewFile *.c,*.cpp,*.cu,*.h,*.hpp set expandtab
    au BufRead,BufNewFile *.c,*.cpp,*.cu,*.h,*.hpp set softtabstop=2  "Insert 2 spaces when tab is pressed
    au BufRead,BufNewFile *.c,*.cpp,*.cu,*.h,*.hpp set smarttab       "Indent instead of tab at start of line
    au BufRead,BufNewFile *.c,*.cpp,*.cu,*.h,*.hpp set shiftround     "Round spaces to nearest shiftwidth multiple
augroup end
