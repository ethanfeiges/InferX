"""Wording for the simulated assistant. The test split gets every third entry of each list."""

SMALL_TALK = [
    "Haha that's so funny", "Sounds good, talk soon", "Hope your week is going well!", "lol same here",
    "Ok see you around", "Thanks for checking in", "That movie was wild", "Long day today, so tired",
    "omg you're the best", "Happy Friday!!", "nah I'm good", "Did you see the game?", "ugh Mondays", "true true",
    "Miss you guys", "Sorry, just saw this", "That's hilarious, send more", "Good morning!", "yep", "no worries at all",
    "Sleep well", "The weather is so nice today", "I love that song", "Wish I was on holiday", "You're so right",
    "Congrats again, so happy for you", "Traffic was awful this morning", "my dog is being so silly", "Same old, same old",
    "What a week", "hahaha stop", "Fair enough",
]
ADS = [
    "50% off everything this weekend only at {store}!", "{store}: your exclusive coupon expires tonight.",
    "New arrivals at {store}. Tap to shop now.", "Flash sale at {store}, free shipping today.",
    "You left something in your cart at {store}", "{store} picks you might like", "Members save 20% at {store} until Sunday",
    "Don't miss out: {store} clearance starts now", "Earn double points at {store} this month",
    "{store} gift cards make great presents", "Last chance! {store} summer sale ends soon",
    "Refer a friend to {store} and get $10",
]
SYSTEM_NOISE = [
    "Rain expected this afternoon in your area.", "Your phone storage is almost full.",
    "Software update available for your device.", "Your weekly screen time report is ready.",
    "Wi-Fi network available nearby.", "45 photos are ready to review in your gallery.", "Your backup finished successfully.",
    "Bluetooth headset disconnected.", "Low battery: 10 percent remaining.", "Night mode will start at 22:00.",
    "2 apps are using location in the background.", "Air quality is moderate today.", "Your step goal is almost done, keep going",
    "Alarm set for 6:45", "Roaming is turned off", "New wallpaper collection available",
]
SURVEYS = [
    "Rate your recent delivery experience in 30 seconds.", "How was your order? Leave a review and earn points.",
    "We'd love your feedback on your last purchase.", "Tell us how we did and enter our prize draw.",
    "Was your package on time? Answer one question.", "Your opinion matters: review your item.",
]
CODES = [
    "Your verification code is {code}. Do not share it.", "{code} is your login code.", "Use code {code} to sign in.",
    "Your one-time password is {code}.", "Security code: {code}. It expires soon.", "Enter {code} to confirm it's you.",
    "PIN for this login: {code}", "{code} is your authentication number", "Do not give this to anyone: {code}",
    "Your sign-in code is {code}, valid for 5 minutes", "Confirm your payment with {code}", "Passcode {code} for your new device",
]
T_APPT_NEW = [
    "New event: {thing} on {when} at {time}, {place}.", "Added to calendar: {thing}, {when}, {time} at {place}.",
    "{thing} booked for {when} at {time} ({place}).", "Calendar invite accepted: {thing} at {place}, {when}, {time}.",
    "Scheduled {thing} at {place} for {when}, {time}.", "Your {thing} is confirmed for {when} at {time}, {place}.",
    "{thing}, {when} {time}, {place}", "{when} {time}: {thing} @ {place}", "{thing} - {place} - {when} - {time}",
    "Don't be late: {thing} {when} {time} at {place}", "Appointment with {place} ({thing}), {when} from {time}",
    "{place} has you down for a {thing} on {when}, {time}", "Saved: {thing} {when}, starts {time}, {place}",
    "See {place} for your {thing}, {time} on {when}", "{time} {when} {thing} at {place}",
]
T_APPT_MOVE = [
    "{thing} moved to {when} at {time}.", "Updated event: {thing} is now on {when}, {time}.",
    "{thing} rescheduled for {when} at {time}.", "Time change: {thing} now takes place {when} at {time}.",
    "Your {thing} has been changed to {when}, {time}.", "New date for {thing}: {when} at {time}.",
    "{thing} pushed back to {when}, {time}", "{thing} now {when} {time}", "Change to your {thing}: come {when} at {time} instead",
    "{place} needs to move your {thing}, new slot {when} {time}", "{thing} brought forward to {when} at {time}",
    "Sorry, your {thing} can't go ahead as planned. Next slot is {when}, {time}", "{thing}: new time {time}, {when}",
    "Revised: {thing}, {when}, {time}", "{thing} shifted to {when} ({time})",
]
T_APPT_CANCEL = [
    "{thing} has been cancelled.", "Event removed: {thing}.", "Cancelled: {thing}.", "{thing} is called off.",
    "Your {thing} was cancelled by {place}.", "{thing} deleted from calendar.", "{thing} won't be happening now",
    "{place} had to scrap your {thing}", "No more {thing}, it's been dropped", "{thing} is off",
    "Unfortunately {place} cancelled your {thing}", "{thing} removed, organiser cancelled",
]
T_APPT_REMIND = [
    "Reminder: {thing} coming up.", "Don't forget your {thing}.", "Upcoming: {thing}.", "{thing} starts in 1 hour",
    "In 15 minutes: {thing}", "Tomorrow: {thing}", "Time to leave for your {thing}", "{thing} is today",
    "Alert set for {thing}",
]
T_TRIP = [
    "Flight {code} to {city} departs {when} at {time}.", "Booked: {code} to {city}, {when}, {time}.",
    "Train to {city}, {when} {time}, seat {seat}.", "{city} trip: {code} leaving {time} on {when}.",
    "Your boarding pass for {city} is ready, {code}, {when} {time}", "{when} {time} {code} to {city}",
    "Ticket to {city} confirmed, {when}, {time} departure", "{code} {city} {when}, gate opens {time}",
    "Coach to {city} leaves {time} on {when}, seat {seat}",
]
T_TRIP_MOVE = [
    "{code} to {city} now leaves {when} at {time}.", "Schedule change: {city} departure moved to {when}, {time}.",
    "Delay notice: your {city} trip now goes at {time} on {when}.", "Rebooked on the {time} to {city}, {when}.",
    "{city} service cancelled, you're on the {time} on {when} instead", "Your {city} journey has changed: {when}, {time}",
    "New departure for {city}: {time}, {when}", "{code} to {city} retimed, {time} {when}", "Trip to {city} shifted to {when} at {time}",
]
T_ORDER = [
    "Order confirmed: {item} from {store}, ${price}.", "Thanks for your order! {item}, ${price}, {store}.",
    "You bought {item} at {store} for ${price}.", "Receipt from {store}: {item}, total ${price}.",
    "{store} order placed for {item}. You paid ${price}.", "Payment of ${price} to {store} for {item}.",
    "${price} at {store}: {item}", "{item} ${price} ({store})", "We've received ${price} for your {item}, thanks from {store}",
    "Card charged ${price}, {store}, {item}", "Purchase complete at {store}: {item}, ${price}", "Hi! {store} got your payment of ${price} for {item}",
    "{store} receipt, {item}, paid ${price}", "You spent ${price} on {item} at {store}", "Bought from {store}: {item} for ${price}",
]
T_SHIPPED = [
    "Your {item} has shipped, arriving {when}.", "{item} is on the way, expected {when}.",
    "Shipping update: {item} due {when}.", "{item} left the warehouse. Delivery {when}.",
    "Good news, {item} ships today and arrives {when}.", "Dispatch notice: {item}, estimated {when}.",
    "{item} out for delivery, by {when}", "Courier has your {item}, tracking says {when}", "{item} in transit, {when}",
    "Your parcel with {item} should reach you {when}", "{item} dispatched, see you {when}", "Tracking update: {item} arrives {when}",
]
T_DELIVERED = [
    "{item} delivered.", "Your {item} has arrived.", "Delivery complete: {item}.", "{item} was left at your door.",
    "Package with {item} was delivered.", "We dropped off your {item}.", "Signed for: {item}", "{item} handed to a neighbour",
    "Parcel received: {item}", "Your {item} is in the safe place you chose", "{item} reached you today", "Courier completed drop-off of {item}",
]
T_REFUND = [
    "Refund issued for {item}.", "{item} returned, refund processed.", "Your {item} refund is complete.",
    "Return accepted: {item} refunded.", "We refunded your {item}.", "Money back for {item} is on its way.",
    "Card credited for the {item} you returned", "Credit note raised: {item}", "{item} sent back, funds returned to your card",
]
T_SUB = [
    "{service} subscription active at ${price} per month.", "Welcome to {service}, ${price} a month.",
    "You're subscribed to {service} for ${price} monthly.", "{service} billing started: ${price} each month.",
    "{service} plan: ${price}/month on your card", "Renews monthly: {service}, ${price}", "{service} membership at ${price} a month",
    "Trial over, {service} now bills you ${price} monthly", "${price} per month for {service} starts today",
]
T_SUB_UP = [
    "{service} is now ${price} per month.", "Price change: {service} goes to ${price} monthly.",
    "Your {service} plan will cost ${price} a month from now on.", "{service} price updated to ${price} per month.",
    "{service} fees rise to ${price} a month next cycle", "Heads up: {service} increases to ${price}/month",
    "New {service} rate of ${price} monthly", "From next bill {service} is ${price} a month", "{service} now charges ${price} monthly",
]
T_BILL = [
    "{biller} bill of ${price} paid.", "${price} paid to {biller} by direct debit.", "{biller}: payment received, ${price}.",
    "Your {biller} bill for this month was ${price}", "${price} taken from your account for {biller}", "{biller} statement settled, ${price}",
    "Paid {biller} ${price}", "Thanks, {biller} got your ${price}", "Autopay: {biller} ${price}",
]
T_ADDRESS = [
    "We just moved! New place is {street}.", "My address is {street} now.", "Send it to {street}, that's where I live now.",
    "Finally settled in at {street}.", "Moved house last week, I'm at {street}.", "Update your contacts: I'm at {street}.",
    "got the keys today, {street} is ours", "new flat is {street} btw", "We're living at {street} from this month",
    "post goes to {street} now", "renting at {street} since Monday", "{street} is home now, come visit",
]
T_ADDRESS_NEXT = [
    "Change of plan, we ended up moving to {street} instead.", "Correction, my address is {street} now.",
    "Moved again, new address is {street}.", "Ignore the old one, I'm at {street} these days.",
    "sorry wrong number on the door, it's {street}", "landlord changed things so we're at {street} now",
    "fell through, we took {street} instead", "scrap that address, use {street}", "actually {street} not the other place",
]
T_PHONE = [
    "New number, it's {phone}.", "My phone number changed to {phone}.", "Use {phone} to reach me from now on.",
    "got a new sim, {phone}", "call me on {phone} now", "changed networks, number is {phone}",
    "{phone} is my new cell", "save this one: {phone}", "my mobile is {phone} these days",
]
T_JOB = [
    "I started as a {role} at {employer}.", "New job! {role} at {employer}.", "Working at {employer} now as a {role}.",
    "just joined {employer}, {role}", "I'm a {role} with {employer} these days", "first week as {role} at {employer} done",
    "took the {role} job at {employer}", "{employer} hired me as a {role}", "now employed at {employer}, {role} role",
]
T_JOB_NEXT = [
    "Left that job, I'm a {role} at {employer} now.", "Switched jobs, {role} with {employer}.",
    "quit, starting at {employer} as a {role}", "moved on, now a {role} at {employer}", "new gig: {role} at {employer}",
    "said goodbye to the old place, {employer} now, {role}", "changed company again, {employer}, {role}",
    "handed in my notice, going to {employer} as {role}", "I work for {employer} now, {role}",
]
T_ALLERGY = [
    "Just so you know I'm allergic to {thing}.", "I am allergic to {thing}, sorry to be a pain.",
    "Careful, I'm allergic to {thing}.", "Quick note, I am allergic to {thing}.", "{thing} make me really ill",
    "can't go near {thing}, bad reaction", "I carry an epipen because of {thing}", "no {thing} for me, doctor's orders",
    "{thing} give me hives", "I react badly to {thing}", "keep {thing} away from me please", "had a {thing} allergy since I was little",
]
T_DIET = [
    "I'm {diet} now, so keep that in mind for meals.", "Going {diet} from now on.", "I've been {diet} for a while.",
    "Heads up that I'm {diet}.", "being {diet} so check the menu", "strictly {diet} these days", "I only eat {diet} food",
    "{diet} since last year, remember", "please make mine {diet}",
]
PREFERENCES = [
    "I always pick the window seat.", "I don't take sugar in my tea.", "We never drive at night.",
    "Mornings are best for me, never book evenings", "I can't stand horror films", "my shoe size is 42 if you're buying",
    "I only drink decaf", "I prefer trains to planes", "Please always call rather than text me", "I hate surprises, tell me first",
    "I do my big shop on Saturdays", "I'm left handed, remember for the seating", "no meetings for me on Fridays ever",
    "I get seasick so no boats", "Sundays are family day for us, always",
]
T_PLAN = [
    "Dinner at {place} on {when}?", "Let's do dinner at {place}, {when}.", "Booked us a table at {place} for {when}.",
    "How about {place} on {when} for dinner?", "Dinner plan: {place}, {when}.", "See you at {place} for dinner, {when}.",
    "dinner {when}? thinking {place}", "got us {place} for dinner {when}", "{place}, {when}, dinner, you in?",
    "Table for dinner at {place} {when}", "fancy {place} {when} for dinner", "let's eat at {place}, dinner {when}",
]
T_PLAN_MOVE = [
    "Actually let's go to {place} instead for dinner.", "Dinner is moved to {place}.", "Change of plan, dinner at {place}.",
    "They're full, so dinner at {place} now.", "New spot for dinner: {place}.", "Switching our dinner to {place}.",
    "first place is shut, dinner at {place} then", "dinner now at {place}, no tables at the other", "scrap that, {place} for dinner",
    "ok {place} works better for dinner", "let's do {place} for dinner after all", "booking fell through, {place} for dinner",
]
T_ERRAND = [
    "Can you collect {kid} from {place} on {when}?", "Please pick up {kid} at {place}, {when}.", "Need you to get {kid} from {place} {when}.",
    "{kid} needs a lift home from {place} {when}", "could you grab {kid} from {place} on {when}", "you're on {kid} duty at {place}, {when}",
    "reminder for us: {kid} at {place} {when}, you collect", "stuck at work, get {kid} from {place} {when}?", "will you fetch {kid} {when}? {place}",
]
T_ERRAND_OFF = [
    "No need to collect {kid} after all.", "Forget picking up {kid}, sorted.", "{kid}'s lift is covered now, don't worry.",
    "never mind about {kid}, my sister will do it", "{kid} is getting the bus so you're off the hook", "cancel the {kid} pickup",
    "{kid} sorted, thanks anyway", "don't need you for {kid} now", "plans changed, {kid} is staying over",
]
