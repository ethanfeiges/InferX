"""Simulated personal-assistant data: calendar, messages, orders and noise on a phone.

No real user data is involved. Because the simulator creates each change itself, it knows the
right memory decision for every item and the current answer to every question.

What this task should remember:
  state:  things that can change and should be replaced when they do, such as an appointment time,
          an order status, a subscription price, a contact's address or a dinner plan.
  event:  things that stay true once they happen, such as a purchase and what it cost.
  core:   stable facts about people, such as allergies and preferences.
  ignore: repeats and noise, such as reminders, ads, small talk and one-time codes.

Samples use LoCoMo's layout, so the same runner and scoring apply, plus a "labels" block keyed by
message id. The test split uses names, items and wording that train and val never see.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from .assistant_wording import (
    ADS, CODES, PREFERENCES, SMALL_TALK, SURVEYS, SYSTEM_NOISE, T_ADDRESS, T_ADDRESS_NEXT, T_ALLERGY, T_APPT_CANCEL,
    T_APPT_MOVE, T_APPT_NEW, T_APPT_REMIND, T_BILL, T_DELIVERED, T_DIET, T_ERRAND, T_ERRAND_OFF, T_JOB, T_JOB_NEXT,
    T_ORDER, T_PHONE, T_PLAN, T_PLAN_MOVE, T_REFUND, T_SHIPPED, T_SUB, T_SUB_UP, T_TRIP, T_TRIP_MOVE,
)

START = date(2025, 3, 3)
DAYS = 28
CATEGORY_DATE, CATEGORY_FACT, CATEGORY_ADVERSARIAL = 2, 4, 5

OWNERS = ["Maya", "Liam", "Noor", "Elena", "Kofi", "Hana", "Marco", "Aisha", "Tomas", "Grace", "Ravi", "Ingrid"]
CONTACTS = ["Sam", "Priya", "Jonas", "Lucia", "Omar", "Beth", "Chen", "Dara", "Felix", "Rosa", "Yusuf", "Nina",
            "Paul", "Mei", "Andre", "Tessa", "Ivan", "Zoe"]
APPOINTMENTS = [("dentist appointment", "Bright Smile Dental"), ("physio session", "Northside Physio"),
                ("car service", "Parkway Garage"), ("haircut", "Studio Nine"), ("vet visit", "Oak Lane Vets"),
                ("eye exam", "ClearView Optics"), ("tax meeting", "Harbor Accounting"),
                ("parent-teacher meeting", "Elmwood School"), ("blood test", "City Clinic"),
                ("boiler inspection", "Warmline Services"), ("skin check", "Riverside Dermatology"),
                ("passport interview", "Central Post Office")]
ITEMS = [("running shoes", "StrideShop", 89), ("desk lamp", "HomeBox", 34), ("coffee grinder", "BrewHouse", 62),
         ("rain jacket", "TrailGear", 118), ("phone case", "CaseCo", 19), ("office chair", "HomeBox", 240),
         ("blender", "KitchenHub", 75), ("backpack", "TrailGear", 58), ("headphones", "SoundMart", 129),
         ("yoga mat", "FitNest", 27), ("toaster", "KitchenHub", 44), ("monitor stand", "DeskWorks", 49)]
SERVICES = [("MusicNow", 11), ("CloudBox storage", 3), ("StreamFlix", 16), ("FitTrack", 9), ("NewsDaily", 7),
            ("GameVault", 13)]
PLACES = ["Luigi's", "the Green Table", "Sakura House", "Cafe Mira", "the Old Mill", "Taverna Eleni", "Blue Door",
          "Casa Roja", "the Harbor Grill"]
STREETS = ["14 Birch Road", "9 Mill Lane", "221 Canal Street", "5 Orchard Close", "78 Station Road", "33 Elm Avenue",
           "60 Lake View", "2 Church Walk", "47 Hill Street"]
ALLERGIES = ["peanuts", "shellfish", "penicillin", "cats", "latex", "pollen"]
DIETS = ["vegetarian", "vegan", "gluten free", "pescatarian"]
CITIES = ["Madrid", "Lisbon", "Oslo", "Prague", "Vienna", "Dublin", "Athens", "Berlin", "Rome"]
BILLERS = ["City Water", "BrightEnergy", "FibreLink", "Metro Council", "HomeGas", "MobileOne", "SafeInsure", "TV Licensing", "Rent Direct"]
ROLES = ["teacher", "paramedic", "electrician", "accountant", "chef", "designer", "pilot", "translator", "plumber"]
EMPLOYERS = ["Northwind", "Halo Labs", "Greenfield School", "Apex Freight", "Riverside Hospital", "Studio Kite", "Dune Foods", "Civic Bank", "Orbit Air"]
KIDS = ["Theo", "Mia", "Jack", "Ella", "Noah", "Zara", "Leo", "Ivy", "Sami"]
PICKUPS = ["school", "football practice", "the dance class", "swim club", "the library", "music lessons", "the youth club", "scouts", "the park"]


def _split_pool(pool: list, split: str) -> list:
    """Test gets every third entry, so its names and wording are never seen in training."""
    return [item for index, item in enumerate(pool) if (index % 3 == 2) == (split == "test")]


def _day(when: date) -> str:
    return f"{when.day} {when:%B %Y}"


class _User:
    def __init__(self, seed: int, split: str, days: int = DAYS, noise_per_day: int = 2):
        self.rng = random.Random(seed)
        self.split = split
        self.days, self.noise_per_day = days, noise_per_day
        self.owner = self.rng.choice(_split_pool(OWNERS, split))
        self.items: list[tuple[int, str, str, dict]] = []
        self.qa: list[dict] = []
        self.sequence = 0

    def pool(self, values: list) -> list:
        shuffled = _split_pool(values, self.split)
        self.rng.shuffle(shuffled)
        return shuffled

    def text(self, templates: list[str], **values) -> str:
        return self.rng.choice(_split_pool(templates, self.split)).format(**values)

    def when(self, day: date) -> str:
        """Dates appear in many formats, while answers always use one."""
        return self.rng.choice([_day(day), f"{day:%a} {day.day} {day:%b}", f"{day:%b} {day.day}", f"{day.day} {day:%b}",
                                f"{day:%A} {day.day} {day:%B}", f"{day.day}/{day.month}/{day.year}"])

    def time(self) -> str:
        return self.rng.choice(["9:30 am", "11:00 am", "2:15 pm", "4:45 pm", "09:30", "16:45", "7pm", "8.15am", "noon"])

    def casual(self, text: str) -> str:
        """People type less carefully than services do."""
        roll = self.rng.random()
        if roll < 0.25:
            text = text.lower().rstrip(".!")
        elif roll < 0.35:
            text = self.rng.choice(["btw ", "fyi ", "oh and ", "also "]) + text[0].lower() + text[1:]
        return text

    def add(self, day: int, speaker: str, text: str, source: str, op: str, kind: str, entity: str = "",
            replaces: str | None = None, core: str | None = None, sensitive: bool = False) -> str:
        """Queue one item and return a handle that resolves to its message id later."""
        handle = f"item{self.sequence}"
        self.sequence += 1
        self.items.append((day, handle, speaker, {
            "text": text, "source": source, "store": op != "ignore", "op": op, "kind": kind, "entity": entity,
            "replaces": replaces, "core": core, "sensitive": sensitive,
        }))
        return handle

    def ask(self, question: str, answer: str, evidence: list[str], category: int, qtype: str, stale: str | None = None):
        self.qa.append({"question": question, "answer": answer, "evidence": evidence, "category": category,
                        "qtype": qtype, "stale_answer": stale})

    def appointments(self, count: int) -> None:
        calendar = f"{self.owner}'s calendar"
        pool = self.pool(APPOINTMENTS)
        count = min(count, len(pool) - 2)  # keep two unused for questions with no answer
        for thing, place in pool[:count]:
            created = self.rng.randrange(self.days - 12)
            first = START + timedelta(days=created + self.rng.randint(14, 25))
            time = self.time()
            current = self.add(created, calendar, self.text(T_APPT_NEW, thing=thing, when=self.when(first), time=time, place=place),
                               "calendar", "add", "state", thing)
            evidence, when, stale = [current], first, None
            fate = self.rng.random()
            if fate < 0.5:
                when = first + timedelta(days=self.rng.randint(1, 6))
                moved = self.add(created + self.rng.randint(1, 7), calendar,
                                 self.text(T_APPT_MOVE, thing=thing, when=self.when(when), time=self.time(), place=place), "calendar", "update",
                                 "state", thing, replaces=current)
                self.ask(f"What date was {self.owner}'s {thing} first set for?", _day(first), [current], CATEGORY_DATE, "history")
                evidence, stale = [moved], _day(first)
            elif fate < 0.65:
                cancelled = self.add(created + self.rng.randint(1, 7), calendar,
                                     self.text(T_APPT_CANCEL, thing=thing, place=place), "calendar", "update", "state",
                                     thing, replaces=current)
                self.ask(f"Is {self.owner}'s {thing} still happening or was it cancelled?", "cancelled", [cancelled],
                         CATEGORY_FACT, "current")
                continue
            self.add(created + self.rng.randint(1, 8), calendar, self.text(T_APPT_REMIND, thing=thing), "calendar", "ignore", "none", thing)
            self.ask(f"When is {self.owner}'s {thing}?", _day(when), evidence, CATEGORY_DATE, "current", stale)
            self.ask(f"Where is {self.owner}'s {thing}?", place, [current], CATEGORY_FACT, "stable")
        for thing, _place in pool[count:count + 2]:
            self.ask(f"When is {self.owner}'s {thing}?", "Not mentioned", [], CATEGORY_ADVERSARIAL, "adversarial")

    def orders(self, count: int) -> None:
        orders = f"{self.owner}'s orders"
        pool = self.pool(ITEMS)
        count = min(count, len(pool) - 2)  # keep two unused for questions with no answer
        for item, store, price in pool[:count]:
            day = self.rng.randrange(self.days - 8)
            placed = self.add(day, orders, self.text(T_ORDER, item=item, store=store, price=price), "orders", "add", "event", item)
            self.ask(f"How much did {self.owner} pay for the {item}?", f"${price}", [placed], CATEGORY_FACT, "stable")
            self.ask(f"Which store did {self.owner} buy the {item} from?", store, [placed], CATEGORY_FACT, "stable")
            eta = START + timedelta(days=day + self.rng.randint(2, 5))
            shipped = self.add(day + 1, orders, self.text(T_SHIPPED, item=item, when=self.when(eta)), "orders", "add", "state", item)
            status, proof, stale = "shipped", shipped, None
            if self.rng.random() < 0.7:
                proof = self.add(day + self.rng.randint(2, 4), orders, self.text(T_DELIVERED, item=item), "orders", "update",
                                 "state", item, replaces=shipped)
                status, stale = "delivered", "shipped"
                if self.rng.random() < 0.3:
                    proof = self.add(day + self.rng.randint(5, 7), orders, self.text(T_REFUND, item=item), "orders", "update",
                                     "state", item, replaces=proof)
                    status, stale = "refunded", "delivered"
            self.ask(f"What is the status of {self.owner}'s {item} order?", status, [proof], CATEGORY_FACT, "current", stale)
        for item, _store, _price in pool[count:count + 2]:
            self.ask(f"How much did {self.owner} pay for the {item}?", "Not mentioned", [], CATEGORY_ADVERSARIAL, "adversarial")

    def subscriptions(self, count: int) -> None:
        orders = f"{self.owner}'s orders"
        for service, price in self.pool(SERVICES)[:count]:
            day = self.rng.randrange(self.days - 10)
            start = self.add(day, orders, self.text(T_SUB, service=service, price=price), "orders", "add", "state", service)
            higher = price + self.rng.randint(1, 4)
            change = self.add(day + self.rng.randint(3, 9), orders, self.text(T_SUB_UP, service=service, price=higher),
                              "orders", "update", "state", service, replaces=start)
            self.ask(f"How much is {self.owner}'s {service} subscription per month now?", f"${higher}", [change],
                     CATEGORY_FACT, "current", f"${price}")

    def contacts(self, names: list[str]) -> None:
        streets = self.pool(STREETS)
        allergies = self.pool(ALLERGIES)
        for index, name in enumerate(names):
            day = self.rng.randrange(self.days - 10)
            first = streets[index % len(streets)]
            current = self.add(day, name, self.casual(self.text(T_ADDRESS, street=first)), "messages", "add", "state", f"{name} address")
            stale = None
            if index % 2 == 0:
                second = streets[(index + 1) % len(streets)]
                current = self.add(day + self.rng.randint(3, 9), name, self.casual(self.text(T_ADDRESS_NEXT, street=second)), "messages",
                                   "update", "state", f"{name} address", replaces=current)
                first, stale = second, first
            self.ask(f"What is {name}'s current address?", first, [current], CATEGORY_FACT, "current", stale)
            allergy = allergies[index % len(allergies)]
            said = self.add(self.rng.randrange(self.days), name, self.casual(self.text(T_ALLERGY, thing=allergy)), "messages", "add", "core",
                            f"{name} allergy", core=f"{name}.allergy")
            self.ask(f"What is {name} allergic to?", allergy, [said], CATEGORY_FACT, "stable")
        self.ask(f"What is {self.owner}'s home address?", "Not mentioned", [], CATEGORY_ADVERSARIAL, "adversarial")

    def plans(self, names: list[str]) -> None:
        places = self.pool(PLACES)
        for index, name in enumerate(names):
            day = self.rng.randrange(self.days - 8)
            when = START + timedelta(days=day + self.rng.randint(5, 12))
            place = places[index * 2 % len(places)]
            current = self.add(day, name, self.casual(self.text(T_PLAN, place=place, when=self.when(when))), "messages", "add", "state", f"{name} dinner")
            changed = places[(index * 2 + 1) % len(places)]
            moved = self.add(day + self.rng.randint(1, 4), name, self.casual(self.text(T_PLAN_MOVE, place=changed)), "messages", "update",
                             "state", f"{name} dinner", replaces=current)
            self.ask(f"Where is {self.owner}'s dinner with {name}?", changed, [moved], CATEGORY_FACT, "current", place)
            self.ask(f"What date is {self.owner}'s dinner with {name}?", _day(when), [current], CATEGORY_DATE, "stable")

    def trips(self, count: int) -> None:
        calendar = f"{self.owner}'s calendar"
        for city in self.pool(CITIES)[:count]:
            day = self.rng.randrange(self.days - 10)
            first = START + timedelta(days=day + self.rng.randint(12, 30))
            code = f"{self.rng.choice(['BA', 'LH', 'FR', 'U2'])}{self.rng.randint(100, 999)}"
            seat = f"{self.rng.randint(1, 60)}{self.rng.choice('ABCD')}"
            booked = self.add(day, calendar, self.text(T_TRIP, code=code, city=city, when=self.when(first), time=self.time(), seat=seat),
                              "calendar", "add", "state", f"{city} trip")
            moved_to = first + timedelta(days=self.rng.randint(1, 3))
            moved = self.add(day + self.rng.randint(1, 8), calendar,
                             self.text(T_TRIP_MOVE, code=code, city=city, when=self.when(moved_to), time=self.time()),
                             "calendar", "update", "state", f"{city} trip", replaces=booked)
            self.ask(f"What date is {self.owner}'s trip to {city}?", _day(moved_to), [moved], CATEGORY_DATE, "current", _day(first))

    def bills(self, count: int) -> None:
        orders = f"{self.owner}'s orders"
        for biller in self.pool(BILLERS)[:count]:
            price = self.rng.randint(15, 180)
            paid = self.add(self.rng.randrange(self.days), orders, self.text(T_BILL, biller=biller, price=price), "orders", "add", "event", biller)
            self.ask(f"How much was {self.owner}'s {biller} bill?", f"${price}", [paid], CATEGORY_FACT, "stable")

    def contact_details(self, names: list[str]) -> None:
        roles, employers = self.pool(ROLES), self.pool(EMPLOYERS)
        for index, name in enumerate(names):
            digits = f"0{self.rng.randint(1000, 9999)} {self.rng.randint(100000, 999999)}"
            said = self.add(self.rng.randrange(self.days), name, self.casual(self.text(T_PHONE, phone=digits)), "messages", "add", "state", f"{name} phone")
            self.ask(f"What is {name}'s phone number?", digits, [said], CATEGORY_FACT, "stable")
            day = self.rng.randrange(self.days - 10)
            employer = employers[index * 2 % len(employers)]
            current = self.add(day, name, self.casual(self.text(T_JOB, role=roles[index % len(roles)], employer=employer)),
                               "messages", "add", "state", f"{name} job")
            stale = None
            if index % 2 == 0:
                stale, employer = employer, employers[(index * 2 + 1) % len(employers)]
                current = self.add(day + self.rng.randint(3, 9), name,
                                   self.casual(self.text(T_JOB_NEXT, role=roles[(index + 1) % len(roles)], employer=employer)),
                                   "messages", "update", "state", f"{name} job", replaces=current)
            self.ask(f"Where does {name} work now?", employer, [current], CATEGORY_FACT, "current", stale)

    def errands(self, names: list[str]) -> None:
        kids, places = self.pool(KIDS), self.pool(PICKUPS)
        for index, name in enumerate(names):
            kid, place = kids[index % len(kids)], places[index % len(places)]
            day = self.rng.randrange(self.days - 6)
            when = START + timedelta(days=day + self.rng.randint(2, 8))
            asked = self.add(day, name, self.casual(self.text(T_ERRAND, kid=kid, place=place, when=self.when(when))), "messages", "add",
                             "state", f"{kid} pickup")
            if index % 2 == 0:
                self.add(day + self.rng.randint(1, 3), name, self.casual(self.text(T_ERRAND_OFF, kid=kid)), "messages", "update",
                         "state", f"{kid} pickup", replaces=asked)

    def preferences(self, names: list[str]) -> None:
        for name, text in zip(names, self.pool(PREFERENCES)):
            self.add(self.rng.randrange(self.days), name, self.casual(text), "messages", "add", "core", f"{name} preference", core=f"{name}.preference")

    def owner_preference(self) -> None:
        diet = self.rng.choice(_split_pool(DIETS, self.split) or DIETS)
        said = self.add(self.rng.randrange(self.days), self.owner, self.casual(self.text(T_DIET, diet=diet)), "messages", "add", "core",
                        f"{self.owner} diet", core=f"{self.owner}.diet")
        self.ask(f"What diet does {self.owner} follow?", diet, [said], CATEGORY_FACT, "stable")

    def noise(self, names: list[str], stores: list[str]) -> None:
        orders = f"{self.owner}'s orders"
        for _ in range(self.days * self.noise_per_day):
            day = self.rng.randrange(self.days)
            kind = self.rng.random()
            if kind < 0.35:
                self.add(day, self.rng.choice(names), self.casual(self.text(SMALL_TALK)), "messages", "ignore", "none")
            elif kind < 0.55:
                store = self.rng.choice(stores)
                self.add(day, f"{store} deals", self.text(ADS, store=store), "promotions", "ignore", "none")
            elif kind < 0.65:
                self.add(day, orders, self.text(SURVEYS), "orders", "ignore", "none")
            elif kind < 0.85:
                self.add(day, "Phone alerts", self.text(SYSTEM_NOISE), "system", "ignore", "none")
            else:
                digits = self.rng.choice([4, 5, 6, 6, 8])
                code = self.rng.randint(10 ** (digits - 1), 10 ** digits - 1)
                speaker = self.rng.choice(["Security", "Bank alerts", self.rng.choice(stores)])
                self.add(day, speaker, self.text(CODES, code=code), "system", "ignore", "none", sensitive=True)

    def sample_long(self, sample_id: str) -> dict:
        """Months of phone activity: every entity the pools allow, and far more noise.

        Storylines are limited by the name pools, so most of the extra length is noise, as on a
        real phone full of notifications.
        """
        names = self.pool(CONTACTS)
        self.appointments(len(APPOINTMENTS))
        self.orders(len(ITEMS))
        self.subscriptions(len(SERVICES))
        self.contacts(names[:5])
        self.plans(names[5:9])
        self.trips(len(CITIES))
        self.bills(len(BILLERS))
        self.contact_details(names[:5])
        self.errands(names[1:5])
        self.preferences([self.owner, names[0], names[5]])
        self.owner_preference()
        self.noise(names[:9], [store for _, store, _ in ITEMS[:6]])
        return self._assemble(sample_id)

    def sample(self, sample_id: str) -> dict:
        names = self.pool(CONTACTS)
        self.appointments(5)
        self.orders(5)
        self.subscriptions(2)
        self.contacts(names[:3])
        self.plans(names[3:5])
        self.trips(2)
        self.bills(3)
        self.contact_details(names[:2])
        self.errands(names[1:3])
        self.preferences([self.owner, names[0], names[3]])
        self.owner_preference()
        self.noise(names[:5], [store for _, store, _ in ITEMS[:6]])
        return self._assemble(sample_id)

    def _assemble(self, sample_id: str) -> dict:
        conversation: dict = {"speaker_a": self.owner, "speaker_b": "assistant"}
        labels, ids = {}, {}
        ordered = sorted(self.items, key=lambda item: (item[0], int(item[1][4:])))
        session = 0
        current_day = None
        for day, handle, speaker, data in ordered:
            if day != current_day:
                session += 1
                current_day = day
                when = START + timedelta(days=day)
                conversation[f"session_{session}_date_time"] = f"8:00 pm on {when.day} {when:%B}, {when.year}"
                conversation[f"session_{session}"] = []
                turn = 0
            turn += 1
            message_id = f"D{session}:{turn}"
            ids[handle] = message_id
            conversation[f"session_{session}"].append({"speaker": speaker, "dia_id": message_id, "text": data["text"]})
            labels[message_id] = {key: value for key, value in data.items() if key != "text"}
        for label in labels.values():
            if label["replaces"]:
                label["replaces"] = ids[label["replaces"]]
        qa = [{**item, "evidence": [ids[handle] for handle in item["evidence"]]} for item in self.qa]
        return {"sample_id": sample_id, "split": self.split, "owner": self.owner, "conversation": conversation,
                "qa": qa, "labels": labels}


def make_user(seed: int, split: str) -> dict:
    return _User(seed, split).sample(f"{split}-{seed}")


def make_long_users(count: int = 3, days: int = 112, noise_per_day: int = 4, seed: int = 0) -> list[dict]:
    """Validation users with four months of history, for testing memory under pressure."""
    return [_User(seed + 4000 + index, "val", days, noise_per_day).sample_long(f"val-long-{seed + 4000 + index}")
            for index in range(count)]


def make_dataset(train: int = 40, val: int = 5, test: int = 10, seed: int = 0) -> dict[str, list[dict]]:
    """Every user has its own seed, and each split gets a separate seed range."""
    return {
        "train": [make_user(seed + 1000 + index, "train") for index in range(train)],
        "val": [make_user(seed + 2000 + index, "val") for index in range(val)],
        "test": [make_user(seed + 3000 + index, "test") for index in range(test)],
    }
