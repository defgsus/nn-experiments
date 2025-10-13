---
tags: [text, ml]
js: ["assets/interest-graph.js"]
style: |
  .interest-graph-word { cursor: pointer; }
  .interest-graph-word:hover { cursor: pointer; text-decoration: underline; }
  .interest-graph-ui-map { 
      width: 400px; 
      height: 400px; 
      position: relative; 
      overflow: hidden;
      background: #181818;
      font-size: .8rem;
  }
  .interest-graph-ui-map-item { font-family: mono: color: rgba(255, 255, 255, .4); }
  .interest-graph-ui-map-item:hover {
      text-shadow: 0 0 7px black;
      color: white;
      z-index: 10;
  }
  .interest-graph-ui-map-item.highlight { 
      color: rgba(255, 255, 255, 1); 
      text-shadow: 0 0 5px white;
      font-weight: bold;    
  }
---

# Interest aggregation

This post might contain off-putting things, for two reasons: Firstly, it is investigating methods of user data aggregation, which can be an offending topic in itself and secondly, it investigates the interests of porn website users. For educational purposes only.

The data has been politely scraped over a couple of days from a free website that hosts images/videos, blogs and public user profiles. For this post i just looked at the "interests", which are zero or more words or groups of words, that each user can specify, to be found more easily on the profile search page, or to simply state the obvious.

The data[[[start
You can download the file [interest-graph.json](interest-graph.json) it contains some 2D-maps of interests and all the edges between interests that have been used together. The format:
```
{
    # all "interests" sorted by most-used first
    "vertices": ["anal", "bdsm", ...],
    # all connections of "interests" [index1, index2, count]
    "edges": [[0,23,3367], [0,5,3278], ...],
    # number of times each "interest" was used
    "vertex_counts": [22339, 10737, ...],
    # list of 2D-maps
    "vertex_positions": [
        {
            "name": "pca300-tsne2",
            # list of xy positions
            "data": [[1.1901,-0.9366], [-0.5495,6.2871], ...]
        },
        ...
    }
}
```
The file has been rendered with [scripts/build_word_relations.py](../../../scripts/build_word_relations.py) but it requires the scraped profiles data, of course.
end]]] published in this article does not contain any direct user relations. Instead it is a point-cloud of interests, gathered by looking at patterns in about 116K profiles. Now dive into the realms of this little porn community[[[start 
If you are worried that what you click and type will be transmitted to *'other parties'*, well.. I did not put anything like that into the javascript ({% for file, url in html.js_map.items() %}<a href="{{url}}" target="_blank">{{file}}</a> {% endfor %}).

That does not mean that your browser can be trusted to keep your web behaviour private. That's not how things are. end]]] and read the technical details below.[[[start 
Haha, sorry. I'm over-using those newly implemented foot notes that work without javascript (thanks to [Tyler Vigen](https://www.tylervigen.com/spurious-correlations) for bringing it to my attention). 

However, this post is not so much a numerical log with endless technical terms and abbreviations like others in this blog. I assumed an audience with a little less technical understanding than usual and describe things in more detail.

In fact this article is more suited for my data blog [defgsus.github.io/blog/](https://defgsus.github.io/blog/) but i just realized that i never want to use Jekyll any more. I tried it back then because it was the '*github*' thing to do but today i rather develop something myself. Which i did right here! So let's go on with this article and don't think about migrating the data blog at some point in distant time...

And yes! I tried to implement recursive integrated foot notes but it's a bit complicated.. end]]]


<div>
<noscript><hr><p>
Enable javascript to browse through the different interests.
<a href="{{ html.js[-1] }}">The script</a> fetches the json data from the static github page and manages the UI, nothing more.
</p><hr></noscript>
<div id="interest-graph-ui" hidden="hidden">
    <input class="interest-graph-ui-input" value="love"/>
    <select class="interest-graph-ui-select"></select>
    <select class="interest-graph-ui-num"></select>
    <div class="flex wrap reverse-when-small" style="margin-top: 1rem;">
        <div class="interest-graph-ui-result"></div>
        <div>&nbsp;</div>
        <div class="interest-graph-ui-map"></div>
    </div>
    <pre class="interest-graph-ui-info"></pre>
</div>
</div>


I do not usually scrape user profiles but this data is genuinely interesting. The particular website attracts and promotes all kinds of [kinkiness](#w=kinky) and [perversions](#w=perversion) as long as it looks like legal content. It's a safe place for people to freely admit they require [small penis humiliation](#w=small%20penis%20humiliation), [cum on food](#w=cum%20on%20food), [cock milking](#w=cock%20milking) or [financial domination](#w=financial%20domination). And i wondered if i could produce a map of internet porn sexual interests.

## Graph/network representation

Those interest strings are case-sensitive in the search page, i guess because in porn language, ANAL means something different than just anal. I lower-cased all strings but otherwise left them as they are.[[[start I also left the encoding errors as they are because i found it hard to fix. For german phrases like [Ärsche](#w=ã„rsche) or [Füße](#w=fã¼ãÿe), it seems like utf8 encoding was latin1-decoded. Reversing that error via `encode("latin1").decode("utf8")` works for stuff that fits into latin1 but there is also a bit of chinese and smileys and other non-latin1 things and i thought, just leave it alone. end]]] [girl](#w=girl) is not [girls](#w=girls) and [mother/son](#w=mother/son) is not the same as [mother son](#w=mother%20son). The site only lists the first 5 pages for any search result and then repeats, which, i think, is a fair thing to do.[[[start It's actually quite funny how it just repeats the same page. That's how you quick-fix things permanently in the not-so-public areas of the web. end]]] I got about 147K profiles at all but included only the ones with at least 5 listed interests. That's still 116K profiles which create a graph of 137K interests and 3.2M interconnections - links between one interest and another.

It turned out that, as graph representation, this data is quite unusable because of the overwhelming number of interconnections. The 30 most often listed interests already have like 600 connections between each other. I tried a few graph/network programs and they would stop reacting within acceptable amounts of time when loading networks with a thousand interests. Not to mention that the layouting algorithms, just by looking at interconnectedness - even weighted, do not have much chance to find a visual arrangement that really helps to understand the interest topology. [anal](#w=anal) is used 23 thousand times and is connected 230 thousands times with almost every other major interest. To visualize a network, you actually need to exclude anal, bdsm and a few others. I also
filtered out edges by some criteria, just to be able to look at the rest of the network more conveniently.

![network of a lot of interests with a lot of edges](assets/interest-graph-network.png)

This network plot shows 375 nodes and many connections have been removed. It's a good starting point but not good for displaying all words and all connections. 

Note that there is nothing much of the users left in this representation, except the sum of how often each interest has been proclaimed together. Sorting the interests-browser above by the [number of edges](#f=edge_count) with your query interest estimates a relation between one interest and another by how often these terms have been used together, but it's kind of boring. The top interests are spread everywhere. 

*Large graphs* are a heavy topic. There exist neat and well-researched algorithms to extract essential information from graphs, like *community detection*. But all of them have a terrible run-time for large graphs (e.g. running for days). Sensible pruning or reduction of graphs is the only means i am aware of to run any of the sophisticated graph algorithms in a short amount of time.[[[start For example, check *A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation* ([arXiv:2402.03358](https://arxiv.org/abs/2402.03358)) end]]]


## Latent representation

There is a faster approach using some kind of *graph representation learning*. We can try to build a model of conglomerated user interest groups from the shared interests. First build a table like this:

|      | user0 | user1 | user... | user100000 |
|------|------:|------:|--------:|-----------:|
| anal |     1 |     0 |       1 |          0 |
| bdsm |     0 |     0 |       1 |          1 |
| ...  |     1 |     1 |       0 |          0 |

Then fit a [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) to the data. It will *squeeze* the vector of a 100K users into something smaller, say 300, while preserving a high explainability of the variances of the data. I used the [sklearn IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html) with a batch size of 300 and constructed above table only for 300 rows at each training step. Otherwise, you'd need a ton of RAM. 

The *components* of the PCA, after fitting, each represent, in decreasing order of variance explanation, an aspect or trade of a modelled user interest group. That compresses the table to 'number-of-interests * 300' numbers which is an acceptable amount of data to process further, while (mathematically proven) preserving a high amount of the interesting stuff.

To limit the size of the json data in this article, i only included interests that are proclaimed at least 30 times, which are about 3,000. [Click here](#n=all) to view them all. This also limits the interest-interest connections to mere 866K.

The interests compressed into numbers by the fitted PCA look like this:

|      | component0 | component1 | component... | component299 |
|-----:|-----------:|-----------:|-------------:|-------------:|
| anal |    113.367 |    15.9827 |     -86.7374 |     0.046496 |
| bdsm |    35.8675 |    42.4413 |      25.5585 |     0.014818 | 
|  ... |    33.5606 |    43.3588 |      23.8518 |     0.206408 |


To conceptually grasp the meaning of this compressed representation, first look at a plot of these numbers for the top-3 interests:

![plot of pca features](assets/interest-pca-features.png)

As mentioned above, the components are sorted by variance explanation. The first component explains the most variance
in the data. So the mean amplitude of these numbers decreases from left to right. Every further component explains a piece of variation that is not explained by the components before. If we would calculate as many components as there are users, the PCA representation would not be a compression and the user interests could be lossless-ly reproduced from the representation. 

Here is a zoomed-in version as bar plot:

![plot of pca features](assets/interest-pca-features-zoomed.png)

We do not yet know what the components actually represent but we can see that each interest is mapped to a unique mixture of these components. For example, the amplitudes of components 10 and 11 are positive for 'bondage' and negative for 'bdsm' and then it flips at component 12. They are mutually exclusive, although bdsm and bondage are quite related interests. These components seem to explain different streams of interests within the modelled bondage group. 'anal' just has a small amplitude in these components so it's not about that. Probably both sub-group users like 'anal', like 23 thousand others.

To represent interests and arrive at these numbers, the PCA is learning a number-of-users-sized vector for each component during training. Those are amplitudes of relations between each component and each user:

|            |       user0 |        user1 |      user... |   user100000 |
|-----------:|------------:|-------------:|-------------:|-------------:|
| component0 |  0.00208772 |   0.00158501 |   0.00215929 |   0.00246174 |
| component1 | -0.00518107 |   0.00306462 |  -0.00334413 |  -0.00128104 |
| component2 |  0.00365257 |    0.0017839 |   0.00269069 | -0.000469027 |
|        ... | -0.00235109 |   -0.0026599 | -0.000812727 |  -0.00101272 |

To calculate the amplitude of, e.g., component 10 from above for the interest 'friends', we create a vector of zeros or ones for each user, putting a one wherever the user has listed 'friends' as interest and then calculate the [dot product](https://en.wikipedia.org/wiki/Dot_product) of this vector and the component vector, which yields a single number: the amplitude of that component for the specified interest.[[[start There is also some mean-shifting and everything is done in big matrix operations but that's not relevant for understanding the concept. end]]] 

As long as we have the internal PCA vectors available, we can map back and forth between interests and users. As an example, we can look at which users are weighted strongest (positively and negatively) by component 10.

| user      |   component 10 weight to user | this user's interests                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:----------|------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| user55363 |                       0.01943 | **[anal](#w=anal)**, [big tits](#w=big%20tits), [bisexuals](#w=bisexuals), [blindfolds](#w=blindfolds), [blondes](#w=blondes), [blowjobs](#w=blowjobs), [bondage](#w=bondage), [brunettes](#w=brunettes), [cock sucking](#w=cock%20sucking), [crossdressers](#w=crossdressers), [cum swapping](#w=cum%20swapping), [dildos](#w=dildos), [fucking](#w=fucking), heels and nylons, [incest](#w=incest), [milfs](#w=milfs), [pussy licking](#w=pussy%20licking), [redheads](#w=redheads), [shemales](#w=shemales), [teen](#w=teen) |
| user83340 |                       0.01595 | [anal sex](#w=anal%20sex), [bbw](#w=bbw), [big ass](#w=big%20ass), [big tits](#w=big%20tits), [blondes](#w=blondes), [blowjobs](#w=blowjobs), [brunettes](#w=brunettes), [deepthroat](#w=deepthroat), [ebony](#w=ebony), [futanari](#w=futanari), [hentai](#w=hentai), **[lesbian](#w=lesbian)**, [masturbation](#w=masturbation), **[milf](#w=milf)**, [petite](#w=petite), pussyfucking, [redheads](#w=redheads), [shemale](#w=shemale), [squirting](#w=squirting), [threesome](#w=threesome)                                 |
| ...       |                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| user48429 |                       -0.0144 | [amateur](#w=amateur), [ass](#w=ass), [bdsm](#w=bdsm), [black](#w=black), [housewives](#w=housewives), **[lesbian](#w=lesbian)**, [lezdom](#w=lezdom), [mature](#w=mature), **[milf](#w=milf)**, [oral](#w=oral), [race play](#w=race%20play), redheads gingers, [role play](#w=role%20play), [schoolgirls](#w=schoolgirls), [small tits](#w=small%20tits), [spanking](#w=spanking), [teens](#w=teens), [young](#w=young)                                                                                                       |
| user353   |                       -0.0150 | **[anal](#w=anal)**, [bareback](#w=bareback), [bdsm](#w=bdsm), [celebs](#w=celebs), [creampies](#w=creampies), [exposure](#w=exposure), [gangbang](#w=gangbang), girl girl, [groupsex](#w=groupsex), [humiliation](#w=humiliation), many more, [mature](#w=mature), **[milf](#w=milf)**, [objects](#w=objects), [orgies](#w=orgies), [pee](#w=pee), [public](#w=public), [teens](#w=teens), [upskirts](#w=upskirts), [voyeur](#w=voyeur)                                                                                        |

The **bold** words are shared between these 4 positive and negative weighted users, all the other words are not shared. The non-linked words are not in the dataset (used less than 30 times) and have therefor not been seen by the PCA. Looking at the interests, it kind of gives a hint of what this component 10 is about. Or does it? For comparison, here is the same for component 0, the most important one:

| user      |   component 0 weight to user | this user's interests                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:----------|-----------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| user239   |                  0.0121141   | [amateur](#w=amateur), [anal](#w=anal), [ass](#w=ass), [bdsm](#w=bdsm), [big tits](#w=big%20tits), [bondage](#w=bondage), [brunettes](#w=brunettes), [cuckold](#w=cuckold), [cum](#w=cum), [fetish](#w=fetish), [humiliation](#w=humiliation), [lingerie](#w=lingerie), [milf](#w=milf), [panties](#w=panties), [pussy](#w=pussy), [sex](#w=sex), [sluts](#w=sluts), [stockings](#w=stockings), [swingers](#w=swingers), [threesome](#w=threesome) |
| user60045 |                  0.0113297   | [amateur](#w=amateur), [anal](#w=anal), [ass](#w=ass), [bbc](#w=bbc), [bdsm](#w=bdsm), [bisexual](#w=bisexual), [blowjobs](#w=blowjobs), [bondage](#w=bondage), [captions](#w=captions), [chubby](#w=chubby), [cock](#w=cock), [creampie](#w=creampie), [crossdressing](#w=crossdressing), [cuckold](#w=cuckold), [curvy](#w=curvy), [interracial](#w=interracial), [latex](#w=latex), [pussy](#w=pussy), [sissy](#w=sissy), [traps](#w=traps)     |
| ...       |                              |                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| user82882 |                 -0.000236861 | bycicling, [clothing](#w=clothing), [cumming](#w=cumming), [dancing](#w=dancing), [drinking](#w=drinking), driving car, [eating](#w=eating), listening to music, [painting](#w=painting), [reading](#w=reading), [riding](#w=riding), [singing](#w=singing), sunbathing (beach or solarium), [swimming](#w=swimming), [walking](#w=walking), watching tv, working, [writing](#w=writing)                                                           |
| user18683 |                 -0.000244663 | [blank](#w=blank), dauerwichsen, eng, [euter](#w=euter), [fett](#w=fett), grabschen, [huren](#w=huren), [hã¤ngetitten](#w=hã¤ngetitten), kein limit, kein taboo, milchtitten, [mutter](#w=mutter), [promis](#w=promis), sacktitten, [schwanger](#w=schwanger), [slips](#w=slips), [spannen](#w=spannen), [strumpfhosen](#w=strumpfhosen), [titten](#w=titten), wã¤schewichsen                                                                      |

I would argue that component 0 has a lot to do with the usage frequency of the interests and, in extension, because of the source material, how 'generically porn' the interest is. user82882 has more facebook-style of interests and, as painful as it is to read user18683's interests, i think they are likely not judged by the PCA as belonging to the "generic porn interests" group. Again, from what has been put in front of the PCA, it has no idea what the words are. It's only looking at distributions. 

Similarly to the weights-per-user we can look at the interests which have the highest or lowest number for a particular component in their representation.

| interest                        |   component 0 amplitude |
|:--------------------------------|------------------------:|
| [anal](#w=anal)                 |              113.383    |
| [cum](#w=cum)                   |               38.269    |
| [mature](#w=mature)             |               37.2679   |
| [bdsm](#w=bdsm)                 |               35.8655   |
| [bondage](#w=bondage)           |               33.5589   |
| [ass](#w=ass)                   |               30.7253   |
| [teens](#w=teens)               |               30.1847   |
| [milf](#w=milf)                 |               29.5561   |
| [bbw](#w=bbw)                   |               29.4968   |
| [teen](#w=teen)                 |               29.0986   |
| ...                             |                         |
| [fgm](#w=fgm)                   |               -0.93819  |
| [walking](#w=walking)           |               -0.93843  |
| [magic](#w=magic)               |               -0.93888  |
| [running](#w=running)           |               -0.939537 |
| [analverkehr](#w=analverkehr)   |               -0.939963 |
| [bleach](#w=bleach)             |               -0.940087 |
| [being naked](#w=being%20naked) |               -0.941865 |
| [working out](#w=working%20out) |               -0.941959 |
| [gardening](#w=gardening)       |               -0.945923 |
| [singing](#w=singing)           |               -0.952731 |

Haha, singing and gardening! So, yes, the argumentation about component 0 can still be made. It reflects how much an interest is a 'typical porn interest' in this particular dataset. Semantically, [analverkehr](#w=analverkehr) belongs to the 'typical' group but the PCA is grouping it together with all the other german words. Not because it knows german but from the distributions in the dataset.  

The component 10 interests ranking:

| interest                      |   component 10 amplitude |
|:------------------------------|-------------------------:|
| [blondes](#w=blondes)         |                 30.2582  |
| [bondage](#w=bondage)         |                 28.7906  |
| [big tits](#w=big%20tits)     |                 25.748   |
| [incest](#w=incest)           |                 24.9399  |
| [bbw](#w=bbw)                 |                 21.939   |
| [brunettes](#w=brunettes)     |                 13.1696  |
| [redheads](#w=redheads)       |                 12.9896  |
| [teen](#w=teen)               |                 10.1763  |
| [hentai](#w=hentai)           |                  8.01334 |
| [sissy](#w=sissy)             |                  6.20947 |
| ...                           |                          |
| [feet](#w=feet)               |                 -5.05988 |
| [milf](#w=milf)               |                 -5.70122 |
| [ass](#w=ass)                 |                 -6.3833  |
| [cuckold](#w=cuckold)         |                 -9.01084 |
| [humiliation](#w=humiliation) |                 -9.59674 |
| [voyeur](#w=voyeur)           |                 -9.85661 |
| [mature](#w=mature)           |                -20.168   |
| [amateur](#w=amateur)         |                -22.3209  |
| [bdsm](#w=bdsm)               |                -27.7806  |
| [teens](#w=teens)             |                -34.5899  |


Blondes and teens are completely opposite interests in this particular component. As well as bondage and bdsm, as we have seen before. Note that 'teen' and 'teens' are also in the opposite sides of this component. Some deep psychological reason seems to exist that users either identify themselves as interested in [teen](#w=teen) or [teens](#w=teens). The terms have not been used together a single time in this dataset.  

Now, whatever exact explanation behind each component's meaning might exist, the mixture of 300 components should give us a quite diverse map of interest territories.

So, we have moved from comparing interests by their number of connections to comparing interests by their similarity in some [latent space](https://en.wikipedia.org/wiki/Latent_space), also called embedding space, latent features, numeric representations or feature vectors. There are many methods to compress complex data into latent features, including many kinds of neural networks. These methods typically create a list (a vector) of numbers of fixed size that can be compared with classic numeric means. 

The PCA is a very powerful and efficient method to get started.

A recommendation system can now suggest similar interests by calculating a single number from two feature vectors, e.g., the euclidean distance. And it works pretty good. In our example, [mother son](#w=mother son&f=pca300-tsne2) and PCA-300 distance lists a lot of similar interests like mother/son, mom-son, aso. which are rarely or never mentioned together. 'The algorithm' just found them to be similar, even though it does not look at the words.

A little privacy notice[[[start 
For educational purposes i'm creating a map of interests, aggregated over the individual user profiles. If we transpose the initial table at the top and put the interests into the columns, we create a map of users, aggregated over their shared interests. That is what's done every day. To target you with ads that supposedly do not waste your time (that's what the ad buyers are told), to calculate an individual price for your purchase that archives a maximum transfer of money to the shareholders (that's what the shareholders are told) or to algorithmically lower the wages of workers.

[https://pluralistic.net/tag/algorithmic-wage-discrimination/](https://pluralistic.net/tag/algorithmic-wage-discrimination/)
end]]].

Below is a comparison for the top-50 interests. It shows the closest interests in terms of edge count (how often used together) and in terms of distance of feature-vectors. Note that we got rid of the 'anal' popping up everywhere without removing it from the graph or similar destructive measures. (The number in brackets is the edge count between top-interest and the closest interest)

| interest                          | closest by edge count                 | closest by pca distance                        |
|:----------------------------------|:--------------------------------------|:-----------------------------------------------|
| [anal](#w=anal)                   | [oral](#w=oral) (3367x)               | [oral](#w=oral) (3367x)                        |
| [bdsm](#w=bdsm)                   | [bondage](#w=bondage) (2982x)         | [whipping](#w=whipping) (268x)                 |
| [bondage](#w=bondage)             | [bdsm](#w=bdsm) (2982x)               | [gags](#w=gags) (496x)                         |
| [mature](#w=mature)               | [bbw](#w=bbw) (2676x)                 | [granny](#w=granny) (1520x)                    |
| [teens](#w=teens)                 | [anal](#w=anal) (2135x)               | [schoolgirls](#w=schoolgirls) (209x)           |
| [cum](#w=cum)                     | [anal](#w=anal) (3278x)               | [balls](#w=balls) (190x)                       |
| [incest](#w=incest)               | [anal](#w=anal) (1952x)               | [mother](#w=mother) (404x)                     |
| [bbw](#w=bbw)                     | [mature](#w=mature) (2676x)           | [ssbbw](#w=ssbbw) (914x)                       |
| [teen](#w=teen)                   | [anal](#w=anal) (2083x)               | [girl](#w=girl) (155x)                         |
| [milf](#w=milf)                   | [mature](#w=mature) (2676x)           | [cougar](#w=cougar) (264x)                     |
| [ass](#w=ass)                     | [anal](#w=anal) (2643x)               | [butt](#w=butt) (292x)                         |
| [humiliation](#w=humiliation)     | [bdsm](#w=bdsm) (1959x)               | [degradation](#w=degradation) (866x)           |
| [femdom](#w=femdom)               | [humiliation](#w=humiliation) (1789x) | [forced bi](#w=forced%20bi) (301x)             |
| [cuckold](#w=cuckold)             | [bbc](#w=bbc) (1687x)                 | [forced bi](#w=forced%20bi) (185x)             |
| [blondes](#w=blondes)             | [brunettes](#w=brunettes) (1898x)     | [brunettes](#w=brunettes) (1898x)              |
| [amateur](#w=amateur)             | [anal](#w=anal) (1678x)               | [girlfriend](#w=girlfriend) (178x)             |
| [sissy](#w=sissy)                 | [anal](#w=anal) (1895x)               | [faggot](#w=faggot) (268x)                     |
| [pussy](#w=pussy)                 | [anal](#w=anal) (2067x)               | [cunt](#w=cunt) (151x)                         |
| [big tits](#w=big%20tits)         | [anal](#w=anal) (1542x)               | [big asses](#w=big%20asses) (254x)             |
| [interracial](#w=interracial)     | [bbc](#w=bbc) (1936x)                 | [big black cock](#w=big%20black%20cock) (174x) |
| [bbc](#w=bbc)                     | [interracial](#w=interracial) (1936x) | [bwc](#w=bwc) (367x)                           |
| [feet](#w=feet)                   | [anal](#w=anal) (1459x)               | [toes](#w=toes) (658x)                         |
| [panties](#w=panties)             | [anal](#w=anal) (1171x)               | [bras](#w=bras) (316x)                         |
| [oral](#w=oral)                   | [anal](#w=anal) (3367x)               | [vaginal](#w=vaginal) (94x)                    |
| [gangbang](#w=gangbang)           | [anal](#w=anal) (2048x)               | [blowbang](#w=blowbang) (242x)                 |
| [lingerie](#w=lingerie)           | [stockings](#w=stockings) (1216x)     | [bras](#w=bras) (161x)                         |
| [shemale](#w=shemale)             | [anal](#w=anal) (1700x)               | [transexual](#w=transexual) (151x)             |
| [asian](#w=asian)                 | [anal](#w=anal) (1154x)               | [thai](#w=thai) (198x)                         |
| [creampie](#w=creampie)           | [anal](#w=anal) (1746x)               | [ao](#w=ao) (78x)                              |
| [stockings](#w=stockings)         | [pantyhose](#w=pantyhose) (1521x)     | [corsets](#w=corsets) (178x)                   |
| [milfs](#w=milfs)                 | [teens](#w=teens) (1255x)             | [gilfs](#w=gilfs) (293x)                       |
| [piss](#w=piss)                   | [anal](#w=anal) (1503x)               | [shit](#w=shit) (221x)                         |
| [crossdressing](#w=crossdressing) | [sissy](#w=sissy) (1484x)             | [transvestite](#w=transvestite) (136x)         |
| [young](#w=young)                 | [teen](#w=teen) (1450x)               | [tiny](#w=tiny) (168x)                         |
| [hairy](#w=hairy)                 | [mature](#w=mature) (1426x)           | [armpits](#w=armpits) (131x)                   |
| [voyeur](#w=voyeur)               | [amateur](#w=amateur) (1078x)         | [spy](#w=spy) (201x)                           |
| [pantyhose](#w=pantyhose)         | [stockings](#w=stockings) (1521x)     | [tights](#w=tights) (670x)                     |
| [redheads](#w=redheads)           | [blondes](#w=blondes) (1455x)         | [freckles](#w=freckles) (189x)                 |
| [sex](#w=sex)                     | [anal](#w=anal) (1052x)               | [fuck](#w=fuck) (109x)                         |
| [captions](#w=captions)           | [incest](#w=incest) (854x)            | [gifs](#w=gifs) (129x)                         |
| [shemales](#w=shemales)           | [anal](#w=anal) (1127x)               | [trannies](#w=trannies) (160x)                 |
| [cock](#w=cock)                   | [cum](#w=cum) (1578x)                 | [balls](#w=balls) (169x)                       |
| [tits](#w=tits)                   | [ass](#w=ass) (1404x)                 | [cunt](#w=cunt) (101x)                         |
| [bukkake](#w=bukkake)             | [anal](#w=anal) (1285x)               | [gokkun](#w=gokkun) (207x)                     |
| [hentai](#w=hentai)               | [anal](#w=anal) (915x)                | [futa](#w=futa) (187x)                         |
| [lesbian](#w=lesbian)             | [anal](#w=anal) (1112x)               | [lezdom](#w=lezdom) (90x)                      |
| [masturbation](#w=masturbation)   | [anal](#w=anal) (1015x)               | [fingering](#w=fingering) (119x)               |
| [blowjob](#w=blowjob)             | [anal](#w=anal) (1625x)               | [rimjob](#w=rimjob) (99x)                      |
| [chubby](#w=chubby)               | [bbw](#w=bbw) (1686x)                 | [plump](#w=plump) (136x)                       |
| [latex](#w=latex)                 | [bondage](#w=bondage) (1152x)         | [pvc](#w=pvc) (515x)                           |

## Visual representation

Okay, so how do we browse this modelled territory of interests except via endless tables of sorted words? The latent vector of any algorithm has a specific size, in our case 300. That requires 300-dimensional vision and thinking capability! Fortunately, other 3-dimensional beings have developed nice tools for further compressing an N-dimensional vector to a more comprehensible 2d or 3d version. Compressing our 100K users vector into two dimensions sounds like a **very** lossy process for sure, but to make far-too-complex data understandable, a plot
of two or three dimensions can be very informative. It's not that a 300-dimensional problem can not be looked at in 2d, it just takes **many** different angles to do so.

One of the most famous algorithms is tSNE ([wiki](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding), [python](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)). It's there for the exact purpose of mapping any feature space into two or three dimensions and make it look good. You don't have to think about anything[[[start except you have millions of items... then you have to think about what to do with your time until tSNE finishes end]]], not even remember what tSNE stands for. Just put your vectors in and get 2d positions out. If there are clusters in the data, tSNE will make them visible. If your data is uniformly distributed, you will see it in the plot. The thing about a point in the map is not if it's east or north, but how far away it is from other points of interest.

