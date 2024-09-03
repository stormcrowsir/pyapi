import requests


url = "https://learning.oreilly.com/api/v2/search/"

payload = {}
headers = {
  'Cookie': '_abck=FD16569D0B0D1C867F202A0682BA4C82~-1~YAAQ0UvWF1FEDZeRAQAATHSYuAzBJcC5DuZLFd4SMD4Dm3wcKhXNeOBPwRpzzx8KTnbI1NDpbw7icXNoq4ltl5CV0by3NDdDzPZYuiP3eEjSMwIZ0vN2nD7NuOcX3Boi1TGmEkNiPL/Qx8Uyv/kmGnYfCJLaBtVTGe4uRDglxIJ3mCYqh6utr6iVOOBKm3+9rK5p7vNfb7PFXEmxNKp0m/hEqKEzBPNei6evqChOZpzo+oMhWy37HCNwmqdna1LqDHRyPiMwsu4I7L5tGxgVLEZJn9WE0N0wyDzNusrt0baA2xyBOigJeNFQucN4KkqE0qaz5pv8th1+RG18OCHcNUvrtn0vcU0BjMdJEjgN0A2yoUIpFPqG3yfkOkHV6JQJ6KeP1mYzfRQ=~-1~-1~-1; ak_bmsc=1EF6BB8EDC7D0461AD02DFF65DEBC0AA~000000000000000000000000000000~YAAQ0UvWF1JEDZeRAQAATHSYuBiqlmg0dMKDVLzeE0slc4w9sDufnRSEi4CoRwn26SKBk5K4YJ0NMeE9MQ+Pu2g2JD4EAS0n+Mw+XgfGT5PQIbCEkCjcQAdTYhf/aV4MJ6hQgB6aSZcZqScGjsHKlRbe/Hko/SQ4F8xuUMv2tCfJk+LdEHQeUdC/R062tMIv2d21KqRAAod/zaDD0jexccGHrmR6uXgHLClLK9PkRT7EGiHQS6+pEFMH6TmHxDOL+gBLqbNekfByGzdZZ/NcC2heX2x9Vn/j9S8LmB+s5WW7vcXx2FUibqP40Zg8VAG3Z53tYEhmhfWB9g7ZRRotQWm5q7SfWuDzPZPS/IIMMj42CMA208dEjbFRKDYjJXc=; bm_sz=9E525DE7E998571E58DC29FEFEAD8CED~YAAQ0UvWF1NEDZeRAQAATHSYuBgTjs6r/O2xYMeyYOIrJMNrb1q2ngMFN3rVaRrmtlV3KWqjHUNrZw7OvOrnJkR7rs/OhFkhvIfg5ZcXKY7YHCdmH/UV0ezuZBcdgsOlAPFy1rk3f7qR+5QPbsLIylhF4POn2ftNNtXpPRtaXnR0U0yuxrB+M+Ec4ML8Qq8eHo10AXNYMPqwyEcmCZvhK7/woqgMN5O1uxR+kSxDnvPAvJb9O8HvGny88iPkUG4NfiZbOyBEgZQKytYFrzt/JAn5VkAD8/A2z8qh10ADmHbmGDuCFuZG8o47xEIDNQpWQeehVGDt8U6M4TutIamGKrnZSXnop62f+ZsjxVbmyx6i~3556919~3617593; akaalb_LearningALB=~op=learning_oreilly_com_GCP_ALB:learning_oreilly_com_gcp1|~rv=43~m=learning_oreilly_com_gcp1:0|~os=3284f997983d0bd4e10a6b83f3b25a7c~id=89dc04c4eee7e9e995786feca579ee51'
}


def fetch_oreilly(keyword):
    
    response = requests.request("GET", f'{url}?query={keyword}', headers=headers, data=payload)
    response_json = response.json()

    # Access the 'result' property
    return response_json.get('results')