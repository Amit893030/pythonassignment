#Q1. Write a Python program to reverse a string without using any built-in string reversal functions.
def reverse_string(input_str):
    reversed_str = ""
    for char in input_str:
        reversed_str = char + reversed_str
    return reversed_str

# Example :
string = input("Enter a string: ")
reversed_string = reverse_string(string)
print("Reversed string:", reversed_string)

#Q2.Implement a function to check if a given string is a palindrome.
def is_palindrome(input_str):
    # Remove spaces and convert to lowercase
    input_str = input_str.replace(" ", "").lower()

    # Check if the string is equal to its reverse
    return input_str == input_str[::-1]


# Example :
string = input("Enter a string: ")
if is_palindrome(string):
    print("The string is a palindrome.")
else:
    print("The string is not a palindrome.")

#Q3. Write a program to find the largest element in a given list.
def find_largest_element(lst):
    if not lst:
        # Return None if the list is empty
        return None

    largest_element = lst[0]  # Assume the first element is the largest

    for element in lst:
        if element > largest_element:
            largest_element = element

    return largest_element

# Example:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

largest = find_largest_element(numbers)
if largest is not None:
    print("The largest element in the list is:", largest)
else:
    print("The list is empty.")

#Q4.Implement a function to count the occurrence of each element in a list.
def count_elements(lst):
    element_count = {}

    for element in lst:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1

    return element_count

# Example usage:
elements = input("Enter a list of elements (separated by spaces): ").split()

count = count_elements(elements)
print("Element counts:")
for element, occurrence in count.items():
    print(f"{element}: {occurrence}")

#Q5. Write a Python program to find the second largest number in a list
def find_second_largest(lst):
    if len(lst) < 2:
        # Return None if the list has less than 2 elements
        return None

    largest = max(lst[0], lst[1])
    second_largest = min(lst[0], lst[1])

    for i in range(2, len(lst)):
        if lst[i] > largest:
            second_largest = largest
            largest = lst[i]
        elif lst[i] > second_largest:
            second_largest = lst[i]

    return second_largest

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

second_largest = find_second_largest(numbers)
if second_largest is not None:
    print("The second largest number in the list is:", second_largest)
else:
    print("The list has less than 2 elements.")

#Q6.Implement a function to remove duplicate elements from a list.
def remove_duplicates(lst):
    seen = set()
    result = []

    for element in lst:
        if element not in seen:
            result.append(element)
            seen.add(element)

    return result

# Example usage:
elements = input("Enter a list of elements (separated by spaces): ").split()

unique_elements = remove_duplicates(elements)
print("List with duplicates removed:", unique_elements)

#Q7. Write a program to calculate the factorial of a given number.
def factorial(n):
    if n < 0:
        return None
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n+1):
            result *= i
        return result

# Example usage:
number = int(input("Enter a number: "))

factorial_result = factorial(number)
if factorial_result is not None:
    print(f"The factorial of {number} is: {factorial_result}")
else:
    print("Factorial is not defined for negative numbers.")

#Q8. Implement a function to check if a given number is prime.
def is_prime(number):
    if number < 2:
        return False

    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return False

    return True

# Example usage:
number = int(input("Enter a number: "))

if is_prime(number):
    print(f"{number} is a prime number.")
else:
    print(f"{number} is not a prime number.")

#Q9. Write a Python program to sort a list of integers in ascending order
def sort_list(lst):
    return sorted(lst)

# Example usage:
numbers = input("Enter a list of integers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

sorted_numbers = sort_list(numbers)
print("Sorted list:", sorted_numbers)

#Q10.Implement a function to find the sum of all numbers in a list.
def find_sum(lst):
    total = 0
    for num in lst:
        total += num
    return total

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

sum_of_numbers = find_sum(numbers)
print("Sum of numbers:", sum_of_numbers)

#Q11.Write a program to find the common elements between two lists.
def find_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    return list(common_elements)

# Example usage:
elements1 = input("Enter elements of the first list (separated by spaces): ").split()
elements2 = input("Enter elements of the second list (separated by spaces): ").split()

common_elements = find_common_elements(elements1, elements2)
if common_elements:
    print("Common elements:", common_elements)
else:
    print("No common elements found.")

#Q12. Implement a function to check if a given string is an anagram of another string.
def is_anagram(str1, str2):
    # Remove whitespace and convert to lowercase
    str1 = str1.replace(" ", "").lower()
    str2 = str2.replace(" ", "").lower()

    # Check if the sorted strings are equal
    return sorted(str1) == sorted(str2)

# Example usage:
string1 = input("Enter the first string: ")
string2 = input("Enter the second string: ")

if is_anagram(string1, string2):
    print("The strings are anagrams.")
else:
    print("The strings are not anagrams.")

#Q13. Write a Python program to generate all permutations of a given string.
def permutations(string):
    if len(string) <= 1:
        return [string]

    # Get the first character
    first = string[0]

    # Generate permutations of the remaining characters
    remaining = permutations(string[1:])

    # Generate permutations by inserting the first character at all possible positions
    result = []
    for perm in remaining:
        for i in range(len(perm) + 1):
            result.append(perm[:i] + first + perm[i:])

    return result


# Example usage:
input_string = input("Enter a string: ")

perms = permutations(input_string)
print("Permutations:")
for perm in perms:
    print(perm)

#Q14.Implement a function to calculate the Fibonacci sequence up to a given number of terms.
def fibonacci_sequence(n):
    sequence = []
    if n >= 1:
        sequence.append(0)
    if n >= 2:
        sequence.append(1)

    for i in range(2, n):
        next_term = sequence[i - 1] + sequence[i - 2]
        sequence.append(next_term)

    return sequence


# Example usage:
terms = int(input("Enter the number of terms: "))

fib_sequence = fibonacci_sequence(terms)
print("Fibonacci sequence:")
print(fib_sequence)

#Q15. Write a program to find the median of a list of numbers.
def find_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)

    if n % 2 == 1:
        # For odd number of elements, return the middle number
        median = sorted_numbers[n // 2]
    else:
        # For even number of elements, return the average of the two middle numbers
        mid_right = n // 2
        mid_left = mid_right - 1
        median = (sorted_numbers[mid_left] + sorted_numbers[mid_right]) / 2

    return median


# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [float(num) for num in numbers]  # Convert input to a list of floats

median = find_median(numbers)
print("Median:", median)

#Q16. Implement a function to check if a given list is sorted in non-decreasing order.
def is_sorted(lst):
    for i in range(1, len(lst)):
        if lst[i] < lst[i - 1]:
            return False
    return True

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

if is_sorted(numbers):
    print("The list is sorted in non-decreasing order.")
else:
    print("The list is not sorted in non-decreasing order.")

#Q17.Write a Python program to find the intersection of two lists.
def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    return list(intersection)

# Example usage:
elements1 = input("Enter elements of the first list (separated by spaces): ").split()
elements2 = input("Enter elements of the second list (separated by spaces): ").split()

intersection = find_intersection(elements1, elements2)
if intersection:
    print("Intersection:", intersection)
else:
    print("No common elements found.")

#Q18.Implement a function to find the maximum subarray sum in a given list.
def find_maximum_subarray_sum(lst):
    if not lst:
        return 0

    max_sum = current_sum = lst[0]

    for num in lst[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

max_subarray_sum = find_maximum_subarray_sum(numbers)
print("Maximum subarray sum:", max_subarray_sum)

#Q19. Write a program to remove all vowels from a given string
def remove_vowels(string):
    vowels = "aeiouAEIOU"
    without_vowels = ""

    for char in string:
        if char not in vowels:
            without_vowels += char

    return without_vowels


# Example usage:
input_string = input("Enter a string: ")

string_without_vowels = remove_vowels(input_string)
print("String without vowels:", string_without_vowels)

#Q20. Implement a function to reverse the order of words in a given sentence.
def reverse_sentence(sentence):
    words = sentence.split()
    reversed_words = words[::-1]
    reversed_sentence = " ".join(reversed_words)
    return reversed_sentence

# Example usage:
input_sentence = input("Enter a sentence: ")

reversed_sentence = reverse_sentence(input_sentence)
print("Reversed sentence:", reversed_sentence)

#Q21. Write a Python program to check if two strings are anagrams of each other.
def is_anagram(str1, str2):
    # Remove whitespace and convert to lowercase
    str1 = str1.replace(" ", "").lower()
    str2 = str2.replace(" ", "").lower()

    # Check if the sorted strings are equal
    return sorted(str1) == sorted(str2)

# Example usage:
string1 = input("Enter the first string: ")
string2 = input("Enter the second string: ")

if is_anagram(string1, string2):
    print("The strings are anagrams.")
else:
    print("The strings are not anagrams.")

#Q22.Implement a function to find the first non-repeating character in a string.
def find_first_non_repeating_char(string):
    char_count = {}

    # Count the occurrence of each character in the string
    for char in string:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # Find the first non-repeating character
    for char in string:
        if char_count[char] == 1:
            return char

    # Return None if no non-repeating character found
    return None


# Example usage:
input_string = input("Enter a string: ")

first_non_repeating_char = find_first_non_repeating_char(input_string)
if first_non_repeating_char is not None:
    print("First non-repeating character:", first_non_repeating_char)
else:
    print("No non-repeating character found.")

#Q23.Write a program to find the prime factors of a given number.
def find_prime_factors(number):
    prime_factors = []
    divisor = 2

    while divisor <= number:
        if number % divisor == 0:
            prime_factors.append(divisor)
            number = number // divisor
        else:
            divisor += 1

    return prime_factors

# Example usage:
number = int(input("Enter a number: "))

factors = find_prime_factors(number)
if factors:
    print("Prime factors:", factors)
else:
    print("The number has no prime factors.")

#Q24 Implement a function to check if a given number is a power of two.
def is_power_of_two(number):
    if number <= 0:
        return False

    while number > 1:
        if number % 2 != 0:
            return False
        number = number // 2

    return True

# Example usage:
number = int(input("Enter a number: "))

if is_power_of_two(number):
    print("The number is a power of two.")
else:
    print("The number is not a power of two.")

#Q25.Write a Python program to merge two sorted lists into a single sorted list.
def merge_sorted_lists(list1, list2):
    merged_list = []
    i = 0
    j = 0

    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1

    # Append the remaining elements from list1, if any
    while i < len(list1):
        merged_list.append(list1[i])
        i += 1

    # Append the remaining elements from list2, if any
    while j < len(list2):
        merged_list.append(list2[j])
        j += 1

    return merged_list

# Example usage:
list1 = input("Enter elements of the first sorted list (separated by spaces): ").split()
list2 = input("Enter elements of the second sorted list (separated by spaces): ").split()

list1 = [int(num) for num in list1]
list2 = [int(num) for num in list2]

merged_list = merge_sorted_lists(list1, list2)
print("Merged and sorted list:", merged_list)

#Q26. Implement a function to find the mode of a list of numbers.
from collections import Counter

def find_mode(numbers):
    counter = Counter(numbers)
    mode_count = max(counter.values())
    mode = [num for num, count in counter.items() if count == mode_count]
    return mode

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

mode = find_mode(numbers)
if len(mode) == 1:
    print("Mode:", mode[0])
else:
    print("Multiple modes:", mode)

#Q27. Write a program to find the greatest common divisor (GCD) of two numbers.
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Example usage:
num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))

gcd_result = gcd(num1, num2)
print("GCD:", gcd_result)

#Q28. Implement a function to calculate the square root of a given number
def square_root(number, epsilon=1e-6):
    if number < 0:
        raise ValueError("Cannot calculate square root of a negative number.")

    guess = number
    while abs(guess * guess - number) > epsilon:
        guess = (guess + number / guess) / 2

    return guess

# Example usage:
number = float(input("Enter a number: "))

sqrt = square_root(number)
print("Square root:", sqrt)

#Q29. Write a Python program to check if a given string is a valid palindrome ignoring non-alphanumeric characters.
def is_valid_palindrome(string):
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_string = ''.join(char.lower() for char in string if char.isalnum())

    # Check if the cleaned string is equal to its reverse
    return cleaned_string == cleaned_string[::-1]

# Example usage:
input_string = input("Enter a string: ")

if is_valid_palindrome(input_string):
    print("The string is a valid palindrome.")
else:
    print("The string is not a valid palindrome.")

#Q30. Implement a function to find the minimum element in a rotated sorted list.
def find_minimum_rotated(nums):
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            # Minimum element is in the right half
            left = mid + 1
        else:
            # Minimum element is in the left half or at mid
            right = mid

    return nums[left]

# Example usage:
numbers = input("Enter a rotated sorted list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

minimum = find_minimum_rotated(numbers)
print("Minimum element:", minimum)


#Q31. Write a program to find the sum of all even numbers in a list
def sum_even_numbers(numbers):
    sum_even = 0

    for num in numbers:
        if num % 2 == 0:
            sum_even += num

    return sum_even

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

sum_even = sum_even_numbers(numbers)
print("Sum of even numbers:", sum_even)

#Q32.Implement a function to calculate the power of a number using recursion.
def power(base, exponent):
    if exponent == 0:
        return 1
    elif exponent > 0:
        return base * power(base, exponent - 1)
    else:
        return 1 / (base * power(base, -exponent - 1))

# Example usage:
base = float(input("Enter the base: "))
exponent = int(input("Enter the exponent: "))

result = power(base, exponent)
print("Result:", result)

#Q33. Write a Python program to remove duplicates from a list while preserving the order.
def remove_duplicates_preserve_order(lst):
    unique_list = []
    seen = set()

    for item in lst:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)

    return unique_list

# Example usage:
input_list = input("Enter a list of elements (separated by spaces): ").split()
input_list = [int(item) for item in input_list]  # Convert input to a list of integers

result = remove_duplicates_preserve_order(input_list)
print("List with duplicates removed:", result)

#Q34.Implement a function to find the longest common prefix among a list of strings.
def longest_common_prefix(strings):
    if not strings:
        return ""

    prefix = strings[0]

    for string in strings[1:]:
        while not string.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""

    return prefix

# Example usage:
input_strings = input("Enter a list of strings (separated by spaces): ").split()

common_prefix = longest_common_prefix(input_strings)
print("Longest common prefix:", common_prefix)

#Q35. Write a program to check if a given number is a perfect square.
def is_perfect_square(number):
    if number < 0:
        return False

    root = int(number ** 0.5)
    return root * root == number

# Example usage:
number = int(input("Enter a number: "))

if is_perfect_square(number):
    print("The number is a perfect square.")
else:
    print("The number is not a perfect square.")

#Q36. Implement a function to calculate the product of all elements in a list
def calculate_product(numbers):
    product = 1

    for num in numbers:
        product *= num

    return product

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

product = calculate_product(numbers)
print("Product:", product)

#Q37. Write a Python program to reverse the order of
# words in a sentence while preserving the word order.
def reverse_sentence_words(sentence):
    words = sentence.split()
    reversed_words = words[::-1]
    reversed_sentence = ' '.join(reversed_words)
    return reversed_sentence

# Example usage:
input_sentence = input("Enter a sentence: ")

reversed_sentence = reverse_sentence_words(input_sentence)
print("Reversed sentence:", reversed_sentence)

#Q38. Implement a function to find the missing number in a given list of consecutive numbers.
def find_missing_number(numbers):
    n = len(numbers) + 1
    total_sum = (n * (n + 1)) // 2
    actual_sum = sum(numbers)
    missing_number = total_sum - actual_sum
    return missing_number

# Example usage:
numbers = input("Enter a list of consecutive numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

missing = find_missing_number(numbers)
print("Missing number:", missing)

#Q39.Write a program to find the sum of digits of a given number.
def sum_of_digits(number):
    sum_digits = 0

    # Convert the number to a string to iterate over its digits
    for digit in str(number):
        sum_digits += int(digit)

    return sum_digits

# Example usage:
number = int(input("Enter a number: "))

sum_of_digits = sum_of_digits(number)
print("Sum of digits:", sum_of_digits)

#Q40. Implement a function to check if a given string is a valid palindrome considering case sensitivity.
def is_valid_palindrome(string):
    # Remove non-alphanumeric characters
    cleaned_string = ''.join(char.lower() for char in string if char.isalnum())

    # Check if the cleaned string is equal to its reverse
    return cleaned_string == cleaned_string[::-1]

# Example usage:
input_string = input("Enter a string: ")

if is_valid_palindrome(input_string):
    print("The string is a valid palindrome.")
else:
    print("The string is not a valid palindrome.")

#Q41. Write a Python program to find the smallest missing positive integer in a list
def find_smallest_missing_positive(nums):
    n = len(nums)

    # Step 1: Move all positive integers to their correct positions
    for i in range(n):
        while 1 <= nums[i] <= n and nums[i] != nums[nums[i] - 1]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

    # Step 2: Find the first position where the number is not in its correct position
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # If all positions have the correct number, then the smallest missing positive is n + 1
    return n + 1

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

smallest_missing = find_smallest_missing_positive(numbers)
print("Smallest missing positive integer:", smallest_missing)

#Q42. Implement a function to find the longest palindrome substring in a given string
def longest_palindrome_substring(s):
    longest_substring = ""

    for i in range(len(s)):
        # Check for odd-length palindromes
        odd_palindrome = expand_around_center(s, i, i)
        longest_substring = max(longest_substring, odd_palindrome, key=len)

        # Check for even-length palindromes
        even_palindrome = expand_around_center(s, i, i + 1)
        longest_substring = max(longest_substring, even_palindrome, key=len)

    return longest_substring

def expand_around_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return s[left + 1:right]

# Example usage:
input_string = input("Enter a string: ")

longest_palindrome = longest_palindrome_substring(input_string)
print("Longest palindrome substring:", longest_palindrome)

#Q43.Write a program to find the number of occurrences of a given element in a list.
def count_occurrences(lst, element):
    count = 0

    for item in lst:
        if item == element:
            count += 1

    return count

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

element = int(input("Enter the element to count: "))

occurrences = count_occurrences(numbers, element)
print("Number of occurrences:", occurrences)


#Q44 Implement a function to check if a given number is a perfect number.
def is_perfect_number(number):
    if number <= 0:
        return False

    divisor_sum = 0

    for i in range(1, number):
        if number % i == 0:
            divisor_sum += i

    return divisor_sum == number

# Example usage:
number = int(input("Enter a number: "))

if is_perfect_number(number):
    print("The number is a perfect number.")
else:
    print("The number is not a perfect number.")

#Q45.Write a Python program to remove all duplicates from a string.
def remove_duplicates(string):
    unique_chars = ""

    for char in string:
        if char not in unique_chars:
            unique_chars += char

    return unique_chars

# Example usage:
input_string = input("Enter a string: ")

result = remove_duplicates(input_string)
print("String with duplicates removed:", result)

#Q46. Implement a function to find the first missing positive.
def first_missing_positive(nums):
    n = len(nums)

    # Step 1: Move all positive integers to their correct positions
    for i in range(n):
        while 1 <= nums[i] <= n and nums[i] != nums[nums[i] - 1]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

    # Step 2: Find the first position where the number is not in its correct position
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    # If all positions have the correct number, then the first missing positive is n + 1
    return n + 1

# Example usage:
numbers = input("Enter a list of numbers (separated by spaces): ").split()
numbers = [int(num) for num in numbers]  # Convert input to a list of integers

missing_positive = first_missing_positive(numbers)
print("First missing positive:", missing_positive)




