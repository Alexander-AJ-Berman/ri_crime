with open('../data/postcode_data.txt') as f:
    read_lines = f.readlines()
    zip_code_search = read_lines[0]
    read_str = ' '.join(read_lines)
    arr = read_str.split(zip_code_search)[1:]

    zip_code_info = {}

    
    for el in arr:
        chunk = el.split('\n')
    
    
    
    
    
    
    # for el in arr:
    #     temp = el.split('\n')
    #     if len(temp) != 155:
    #         print(len(temp))

    
    
    
    
    
    
    
    # zip_codes = []
    # avg_commute_time = []
    # for i, line in enumerate(arr):
    #     # ZIP CODES
    #     if line == arr[0]:
    #         zip_codes.append(arr[i + 1].rstrip())
    #     # AVERAGE COMMUTE TIME (MINUTES)
    #     if line == arr[153]:
    #         avg_commute_time.append(arr[i + 1].rstrip())
    # print(len(zip_codes))
    # print(len(avg_commute_time))
    