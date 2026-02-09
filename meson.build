project('anna', 'cpp', version : '0.1', default_options : [ 'b_ndebug=if-release' ])

benchmark = dependency('benchmark')
includes = include_directories('include', 'eigen-5.0.0')
page_size = run_command('getconf', 'PAGE_SIZE', check : true).stdout().strip()
add_global_arguments('-DANNA_PAGE_SIZE=' + page_size, language : 'cpp')

subdir('benchmarks')
