def create_col_and_color_lists(df, test_types): 
    '''
    creates a list of all the questionnaire columns in the avgs df, organized by test type
    and creates another list, aligned to the first, that contains color assignments
    '''
    colors = ['r', 'b', 'g', 'y']
    all_test_columns= []
    colors_by_col = []

    for i in range(len(test_types)): 
        test_type=test_types[i]
        current_color = colors[i]

        # get the columns that contain the current test type 
        curr_tests = [col for col in df.columns if test_type in col]
        all_test_columns.extend(curr_tests)
        colors_by_col.extend(current_color * len(curr_tests))
    return all_test_columns, colors_by_col