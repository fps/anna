project('anna', 'cpp', version : '0.1', default_options : [ 'b_ndebug=if-release' ])

benchmark_dep = dependency('benchmark')
gtest_dep = dependency('gtest_main')
sndfile_dep = dependency('sndfile')

includes = include_directories('include', 'eigen-5.0.0')

page_size = run_command('getconf', 'PAGE_SIZE', check : true).stdout().strip()
add_global_arguments('-DANNA_PAGE_SIZE=' + page_size, language : 'cpp')

subdir('tests')
subdir('benchmarks')
