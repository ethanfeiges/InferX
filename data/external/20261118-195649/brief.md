# Brief for the external writer (draft: edit freely)

You are writing test data for a personal assistant that runs on a phone. The assistant reads what
lands on the phone (calendar entries, messages, emails, order and delivery updates, notes, bank and app
notifications) and later answers questions about the owner's life. We want to see whether it keeps up
when details change.

## What to write

Write 5 people. For each person, write their phone's stream over about 2 to 4 months: 80 to 120 items,
in time order, grouped into days.

- Make each person different: age, job, country, family, how they write.
- Mix the sources. Write each item the way that source would, including short, messy or informal texts.
- Over the period, many details should change: plans rescheduled or cancelled, a new address, phone number,
  job, doctor, flight, delivery date, price, password, and so on. Some details change more than once.
  Some changes should be stated indirectly, in a way a person would understand but a keyword search might miss.
- Some items should contain a secret: a password, PIN, one-time code, card or bank number, ID number,
  or API key.
- Include plenty of items that are not worth remembering.
- Other people appear too. Their details should not be confused with the owner's.

## Labels for each item

- `store`: true if a good assistant should remember this item, false if not.
- `kind`: "state" if it says what is currently true and could change later (an appointment time, an address);
  "event" if it says something happened (a purchase, a finished trip). null if `store` is false.
- `replaces`: the `id` of the earlier item this one makes out of date, or null.
- `sensitive`: true if the item contains a secret as listed above.

## Questions for each person

Write 25 to 40 questions someone might later ask the assistant, with the correct answer as of the end of the
period and the `id`s of the items that support it. For a question about a detail that changed, also give the
old answer in `stale_answer`. Include a few questions about earlier versions ("What was the dentist appointment
first set for?") and a few that cannot be answered from the items (answer: "not mentioned").

## Output

One JSON file per person, named person_1.json to person_5.json:

```json
{
  "owner": "<first name>",
  "items": [
    {"id": "<unique>", "day": "<YYYY-MM-DD>", "time": "<HH:MM>", "source": "<calendar | message | email | order | note | bank | app>",
     "speaker": "<who or what it comes from>", "text": "<the item>",
     "store": true, "kind": "state", "replaces": null, "sensitive": false}
  ],
  "questions": [
    {"question": "...", "answer": "...", "evidence": ["<id>"], "stale_answer": null, "type": "<current | history | stable | unanswerable>"}
  ]
}
```

Write everything yourself. Do not use or look for any existing files or datasets.
