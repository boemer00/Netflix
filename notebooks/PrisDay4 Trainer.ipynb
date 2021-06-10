{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "581f9b25",
   "metadata": {},
   "source": [
    "# Trainer Class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "821dcdf2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.compose import ColumnTransformer\n",
    "#from sklearn import set_config; set_config(ddisplay='diagram')\n",
    "import math\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "c53695c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>avg_review_score</th>\n",
       "      <th>n_reviews</th>\n",
       "      <th>year</th>\n",
       "      <th>title</th>\n",
       "      <th>Title</th>\n",
       "      <th>Year</th>\n",
       "      <th>Rated</th>\n",
       "      <th>Released</th>\n",
       "      <th>Runtime</th>\n",
       "      <th>Genre</th>\n",
       "      <th>...</th>\n",
       "      <th>Response</th>\n",
       "      <th>Internet Movie Database</th>\n",
       "      <th>Index_match</th>\n",
       "      <th>DVD</th>\n",
       "      <th>BoxOffice</th>\n",
       "      <th>Production</th>\n",
       "      <th>Website</th>\n",
       "      <th>Rotten Tomatoes</th>\n",
       "      <th>Metacritic</th>\n",
       "      <th>Ratings</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3.749543</td>\n",
       "      <td>547</td>\n",
       "      <td>2003</td>\n",
       "      <td>Dinosaur Planet</td>\n",
       "      <td>Dinosaur Planet</td>\n",
       "      <td>2003</td>\n",
       "      <td>NaN</td>\n",
       "      <td>14-Dec-03</td>\n",
       "      <td>50 min</td>\n",
       "      <td>Documentary, Animation, Family</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>7.7/10</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.641153</td>\n",
       "      <td>2012</td>\n",
       "      <td>1997</td>\n",
       "      <td>Character</td>\n",
       "      <td>Character</td>\n",
       "      <td>1997</td>\n",
       "      <td>R</td>\n",
       "      <td>27-Mar-98</td>\n",
       "      <td>122 min</td>\n",
       "      <td>Crime, Drama, Mystery</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>7.7/10</td>\n",
       "      <td>3</td>\n",
       "      <td>04-Feb-03</td>\n",
       "      <td>$623,983</td>\n",
       "      <td>Almerica Film</td>\n",
       "      <td>NaN</td>\n",
       "      <td>92%</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.084396</td>\n",
       "      <td>1019</td>\n",
       "      <td>1997</td>\n",
       "      <td>Sick</td>\n",
       "      <td>Sick</td>\n",
       "      <td>1997</td>\n",
       "      <td>Not Rated</td>\n",
       "      <td>07-Nov-97</td>\n",
       "      <td>90 min</td>\n",
       "      <td>Documentary</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>7.5/10</td>\n",
       "      <td>6</td>\n",
       "      <td>15-Feb-17</td>\n",
       "      <td>$116,806</td>\n",
       "      <td>Sick-the Life and Death of Bob Flanagan-Superm...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>91%</td>\n",
       "      <td>82/100</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.129032</td>\n",
       "      <td>93</td>\n",
       "      <td>1992</td>\n",
       "      <td>8 Man</td>\n",
       "      <td>8 Man</td>\n",
       "      <td>1992</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>83 min</td>\n",
       "      <td>Action, Sci-Fi</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>5.4/10</td>\n",
       "      <td>7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.417582</td>\n",
       "      <td>546</td>\n",
       "      <td>1947</td>\n",
       "      <td>My Favorite Brunette</td>\n",
       "      <td>My Favorite Brunette</td>\n",
       "      <td>1947</td>\n",
       "      <td>Passed</td>\n",
       "      <td>04-Apr-47</td>\n",
       "      <td>87 min</td>\n",
       "      <td>Comedy, Crime, Mystery, Romance, Thriller</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>6.8/10</td>\n",
       "      <td>12</td>\n",
       "      <td>10-Mar-16</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Paramount Pictures, Hope Enterprises</td>\n",
       "      <td>NaN</td>\n",
       "      <td>75%</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10381</th>\n",
       "      <td>3.411855</td>\n",
       "      <td>1957</td>\n",
       "      <td>1978</td>\n",
       "      <td>Interiors</td>\n",
       "      <td>Interiors</td>\n",
       "      <td>1978</td>\n",
       "      <td>PG</td>\n",
       "      <td>06-Oct-78</td>\n",
       "      <td>92 min</td>\n",
       "      <td>Drama</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>7.4/10</td>\n",
       "      <td>17763</td>\n",
       "      <td>02-Feb-17</td>\n",
       "      <td>$10,432,366</td>\n",
       "      <td>Rollins-Joffe Productions</td>\n",
       "      <td>NaN</td>\n",
       "      <td>79%</td>\n",
       "      <td>67/100</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10382</th>\n",
       "      <td>3.867112</td>\n",
       "      <td>64957</td>\n",
       "      <td>1998</td>\n",
       "      <td>Shakespeare in Love</td>\n",
       "      <td>Shakespeare in Love</td>\n",
       "      <td>1998</td>\n",
       "      <td>R</td>\n",
       "      <td>08-Jan-99</td>\n",
       "      <td>123 min</td>\n",
       "      <td>Comedy, Drama, History, Romance</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>7.1/10</td>\n",
       "      <td>17764</td>\n",
       "      <td>21-Apr-16</td>\n",
       "      <td>$100,317,794</td>\n",
       "      <td>Miramax Films, Bedford Falls Productions, Univ...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>92%</td>\n",
       "      <td>87/100</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10383</th>\n",
       "      <td>2.839207</td>\n",
       "      <td>1362</td>\n",
       "      <td>2000</td>\n",
       "      <td>Epoch</td>\n",
       "      <td>Epoch</td>\n",
       "      <td>2001</td>\n",
       "      <td>PG-13</td>\n",
       "      <td>24-Nov-01</td>\n",
       "      <td>96 min</td>\n",
       "      <td>Sci-Fi, Thriller</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>5.0/10</td>\n",
       "      <td>17768</td>\n",
       "      <td>25-Jan-17</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Metro-Goldwyn-Mayer</td>\n",
       "      <td>NaN</td>\n",
       "      <td>16%</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10384</th>\n",
       "      <td>2.498592</td>\n",
       "      <td>6749</td>\n",
       "      <td>2003</td>\n",
       "      <td>The Company</td>\n",
       "      <td>The Company</td>\n",
       "      <td>2003</td>\n",
       "      <td>PG-13</td>\n",
       "      <td>20-May-04</td>\n",
       "      <td>112 min</td>\n",
       "      <td>Drama, Music, Romance</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>6.3/10</td>\n",
       "      <td>17769</td>\n",
       "      <td>16-Apr-12</td>\n",
       "      <td>$2,283,914</td>\n",
       "      <td>First Snow Production, Capitol Films, Sandcast...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>72%</td>\n",
       "      <td>73/100</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10385</th>\n",
       "      <td>2.816504</td>\n",
       "      <td>921</td>\n",
       "      <td>2003</td>\n",
       "      <td>Alien Hunter</td>\n",
       "      <td>Alien Hunter</td>\n",
       "      <td>2003</td>\n",
       "      <td>R</td>\n",
       "      <td>19-Jul-03</td>\n",
       "      <td>92 min</td>\n",
       "      <td>Action, Adventure, Sci-Fi</td>\n",
       "      <td>...</td>\n",
       "      <td>True</td>\n",
       "      <td>5.1/10</td>\n",
       "      <td>17770</td>\n",
       "      <td>20-Apr-13</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>16%</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10386 rows × 34 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       avg_review_score  n_reviews  year                 title  \\\n",
       "0              3.749543        547  2003       Dinosaur Planet   \n",
       "1              3.641153       2012  1997             Character   \n",
       "2              3.084396       1019  1997                  Sick   \n",
       "3              2.129032         93  1992                 8 Man   \n",
       "4              3.417582        546  1947  My Favorite Brunette   \n",
       "...                 ...        ...   ...                   ...   \n",
       "10381          3.411855       1957  1978             Interiors   \n",
       "10382          3.867112      64957  1998   Shakespeare in Love   \n",
       "10383          2.839207       1362  2000                 Epoch   \n",
       "10384          2.498592       6749  2003           The Company   \n",
       "10385          2.816504        921  2003          Alien Hunter   \n",
       "\n",
       "                      Title  Year      Rated   Released  Runtime  \\\n",
       "0           Dinosaur Planet  2003        NaN  14-Dec-03   50 min   \n",
       "1                 Character  1997          R  27-Mar-98  122 min   \n",
       "2                      Sick  1997  Not Rated  07-Nov-97   90 min   \n",
       "3                     8 Man  1992        NaN        NaN   83 min   \n",
       "4      My Favorite Brunette  1947     Passed  04-Apr-47   87 min   \n",
       "...                     ...   ...        ...        ...      ...   \n",
       "10381             Interiors  1978         PG  06-Oct-78   92 min   \n",
       "10382   Shakespeare in Love  1998          R  08-Jan-99  123 min   \n",
       "10383                 Epoch  2001      PG-13  24-Nov-01   96 min   \n",
       "10384           The Company  2003      PG-13  20-May-04  112 min   \n",
       "10385          Alien Hunter  2003          R  19-Jul-03   92 min   \n",
       "\n",
       "                                           Genre  ... Response  \\\n",
       "0                 Documentary, Animation, Family  ...     True   \n",
       "1                          Crime, Drama, Mystery  ...     True   \n",
       "2                                    Documentary  ...     True   \n",
       "3                                 Action, Sci-Fi  ...     True   \n",
       "4      Comedy, Crime, Mystery, Romance, Thriller  ...     True   \n",
       "...                                          ...  ...      ...   \n",
       "10381                                      Drama  ...     True   \n",
       "10382            Comedy, Drama, History, Romance  ...     True   \n",
       "10383                           Sci-Fi, Thriller  ...     True   \n",
       "10384                      Drama, Music, Romance  ...     True   \n",
       "10385                  Action, Adventure, Sci-Fi  ...     True   \n",
       "\n",
       "      Internet Movie Database Index_match        DVD     BoxOffice  \\\n",
       "0                      7.7/10           1        NaN           NaN   \n",
       "1                      7.7/10           3  04-Feb-03      $623,983   \n",
       "2                      7.5/10           6  15-Feb-17      $116,806   \n",
       "3                      5.4/10           7        NaN           NaN   \n",
       "4                      6.8/10          12  10-Mar-16           NaN   \n",
       "...                       ...         ...        ...           ...   \n",
       "10381                  7.4/10       17763  02-Feb-17   $10,432,366   \n",
       "10382                  7.1/10       17764  21-Apr-16  $100,317,794   \n",
       "10383                  5.0/10       17768  25-Jan-17           NaN   \n",
       "10384                  6.3/10       17769  16-Apr-12    $2,283,914   \n",
       "10385                  5.1/10       17770  20-Apr-13           NaN   \n",
       "\n",
       "                                              Production Website  \\\n",
       "0                                                    NaN     NaN   \n",
       "1                                          Almerica Film     NaN   \n",
       "2      Sick-the Life and Death of Bob Flanagan-Superm...     NaN   \n",
       "3                                                    NaN     NaN   \n",
       "4                   Paramount Pictures, Hope Enterprises     NaN   \n",
       "...                                                  ...     ...   \n",
       "10381                          Rollins-Joffe Productions     NaN   \n",
       "10382  Miramax Films, Bedford Falls Productions, Univ...     NaN   \n",
       "10383                                Metro-Goldwyn-Mayer     NaN   \n",
       "10384  First Snow Production, Capitol Films, Sandcast...     NaN   \n",
       "10385                                                NaN     NaN   \n",
       "\n",
       "      Rotten Tomatoes  Metacritic  Ratings  \n",
       "0                 NaN         NaN      NaN  \n",
       "1                 92%         NaN      NaN  \n",
       "2                 91%      82/100      NaN  \n",
       "3                 NaN         NaN      NaN  \n",
       "4                 75%         NaN      NaN  \n",
       "...               ...         ...      ...  \n",
       "10381             79%      67/100      NaN  \n",
       "10382             92%      87/100      NaN  \n",
       "10383             16%         NaN      NaN  \n",
       "10384             72%      73/100      NaN  \n",
       "10385             16%         NaN      NaN  \n",
       "\n",
       "[10386 rows x 34 columns]"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"./../raw_data/merged_movies_by_index (1).csv\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "c4274777",
   "metadata": {},
   "outputs": [],
   "source": [
    "def data_wrangling(df):\n",
    "    \"\"\" cleaning irrelevant rows and columns \"\"\" \n",
    "    # drop irrelevant columns\n",
    "    df = df.drop(columns=['title', 'year', 'Awards', 'Poster', 'Metascore', 'DVD',\n",
    "                          'BoxOffice', 'Internet Movie Database','totalSeasons',\n",
    "                          'imdbVotes','Website', 'Response', 'Production', 'Metacritic', 'Ratings'])\n",
    "    ## fill nan and' min', convert to int and replace zero for the mean\n",
    "    df['Runtime'] = df['Runtime'].fillna(0).apply(lambda x: str(x).replace(',', ''))\n",
    "    df['Runtime'] = df['Runtime'].apply(lambda x: float(str(x).replace(' min', '')))\n",
    "    df['Runtime'] = df['Runtime'].replace(0, df['Runtime'].mean())\n",
    "    ## fill nan and remove '%', convert to float and replace zero for the mean\n",
    "    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].fillna(0) \n",
    "    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].apply(lambda x: float(str(x).replace('%', '')))\n",
    "    df['Rotten Tomatoes'] = df['Rotten Tomatoes'].replace(0, df['Rotten Tomatoes'].mean())\n",
    "    ## replace countries and genre with most frequent values\n",
    "    freq_country = df[['Country']].value_counts().reset_index()['Country'][0]\n",
    "    df['Country'] = df['Country'].replace(0, freq_country).replace('United States', freq_country)\n",
    "    freq_genre = df['Genre'].mode()[0]\n",
    "    df['Genre'] = df['Genre'].replace(np.nan, freq_genre)\n",
    "    # replace null values with unknown\n",
    "    df['Actors'] = df['Actors'].replace(np.nan,'unknown')\n",
    "    df['Director'] = df['Director'].replace(np.nan,'unknown')\n",
    "    df['Writer'] = df['Writer'].replace(np.nan,'unknown')\n",
    "    df['Plot'] = df['Plot'].replace(np.nan,'unknown')\n",
    "    ## Language binary (either contains English or not)\n",
    "    df[\"Language\"] = df[[\"Language\"]].fillna(\"English\")\n",
    "    def language_binary(x):\n",
    "        if x.find(\"English\") != -1:\n",
    "            return \"English Available\"\n",
    "        else:\n",
    "            return \"English N/A\"\n",
    "    df[\"Language_binary\"] = df[\"Language\"].map(language_binary)\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "04c8cfd6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "movie     9909\n",
       "series     477\n",
       "Name: Type, dtype: int64"
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Type'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "87df63eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "90.0     444\n",
       "95.0     317\n",
       "100.0    298\n",
       "93.0     277\n",
       "97.0     268\n",
       "        ... \n",
       "282.0      1\n",
       "390.0      1\n",
       "11.0       1\n",
       "187.0      1\n",
       "600.0      1\n",
       "Name: Runtime, Length: 288, dtype: int64"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Runtime'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "id": "25d646ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "R            3542\n",
       "PG-13        1252\n",
       "Not Rated    1242\n",
       "PG           1172\n",
       "Approved      435\n",
       "G             325\n",
       "Unrated       294\n",
       "Passed        200\n",
       "TV-PG         132\n",
       "TV-14         129\n",
       "TV-G           57\n",
       "TV-MA          56\n",
       "TV-Y7          29\n",
       "GP             22\n",
       "TV-Y           20\n",
       "NC-17          17\n",
       "X              13\n",
       "NOT RATED      13\n",
       "M              12\n",
       "UNRATED         9\n",
       "M/PG            6\n",
       "E               2\n",
       "APPROVED        2\n",
       "TV-Y7-FV        2\n",
       "TV-13           1\n",
       "Name: Rated, dtype: int64"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Rated'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "ee4b2b39",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = data_wrangling(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "d3b56df8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#df[\"Runtime\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "cb691a47",
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df[\"avg_review_score\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "7a4ba4fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df[[\"Year\", \"Rated\",\"Language_binary\", \"Runtime\"]] #,"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "id": "c72eb619",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Trainer(object):\n",
    "    def __init__(self, X, y):\n",
    "        \"\"\"\n",
    "            X: pandas DataFrame\n",
    "            y: pandas Series\n",
    "        \"\"\"\n",
    "        self.pipeline = None\n",
    "        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3)\n",
    "        \n",
    "    def set_pipeline(self):\n",
    "        \"\"\"defines the pipeline as a class attribute\"\"\"\n",
    "        # Impute then Scale for numerical variables: \n",
    "        num_transformer = Pipeline([\n",
    "        ('imputer', SimpleImputer(fill_value ='nan')),\n",
    "        ('scaler', StandardScaler())])\n",
    "\n",
    "        # Encode categorical variables\n",
    "        cat_transformer = Pipeline([\n",
    "        ('imputer', SimpleImputer(fill_value ='nan', strategy='constant')),\n",
    "        ('OHO', OneHotEncoder(handle_unknown='ignore', sparse = False))])\n",
    "\n",
    "        # Paralellize \"num_transformer\" and \"One hot encoder\"\n",
    "        preprocessor = ColumnTransformer([\n",
    "            ('num_transformer', num_transformer, ['Year','Runtime']),\n",
    "            ('cat_transformer', cat_transformer, ['Rated', 'Language_binary'])],\n",
    "        remainder='passthrough')\n",
    "\n",
    "        self.pipeline = Pipeline([\n",
    "                ('preprocessor', preprocessor),\n",
    "                ('linear_model', LinearRegression())\n",
    "            ])\n",
    "    def fit_pipeline(self):\n",
    "        self.pipeline = self.pipeline.fit(self.X_train, self.y_train)\n",
    "        print('pipeline fitted')\n",
    "        \n",
    "    def evaluate(self):\n",
    "        y_pred_train = self.pipeline.predict(self.X_train)\n",
    "        mse_train = mean_squared_error(self.y_train, y_pred_train)\n",
    "        rmse_train = np.sqrt(mse_train)\n",
    "        \n",
    "        y_pred_test = self.pipeline.predict(self.X_test)\n",
    "        mse_test = mean_squared_error(self.y_test, y_pred_test)\n",
    "        rmse_test = np.sqrt(mse_test)\n",
    "        return (round(rmse_train, 3) ,round(rmse_test, 3))\n",
    "        \n",
    "    def predict(self, X):\n",
    "        y_pred = self.pipeline.predict(X)\n",
    "        return y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "80020d28",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<__main__.Trainer at 0x1469d00a0>"
      ]
     },
     "execution_count": 127,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Trainer(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "e766be14",
   "metadata": {},
   "outputs": [],
   "source": [
    "trainer = Trainer(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "74e73c2b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(7270, 4)\n",
      "(3116, 4)\n"
     ]
    }
   ],
   "source": [
    "print(trainer.X_train.shape)\n",
    "print(trainer.X_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "af378d6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "trainer.set_pipeline()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "9f61e236",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "pipeline fitted\n"
     ]
    }
   ],
   "source": [
    "trainer.fit_pipeline()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "2cc01bb1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.461, 0.468)"
      ]
     },
     "execution_count": 117,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer.evaluate()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "bcf45133",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method Trainer.fit_pipeline of <__main__.Trainer object at 0x1469d07f0>>"
      ]
     },
     "execution_count": 118,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer.fit_pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "41aef328",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "         steps=[('preprocessor',\n",
       "                 ColumnTransformer(n_jobs=None, remainder='passthrough',\n",
       "                                   sparse_threshold=0.3,\n",
       "                                   transformer_weights=None,\n",
       "                                   transformers=[('num_transformer',\n",
       "                                                  Pipeline(memory=None,\n",
       "                                                           steps=[('imputer',\n",
       "                                                                   SimpleImputer(add_indicator=False,\n",
       "                                                                                 copy=True,\n",
       "                                                                                 fill_value='nan',\n",
       "                                                                                 missing_values=nan,\n",
       "                                                                                 strategy='mean',\n",
       "                                                                                 verbose=0)),\n",
       "                                                                  ('scaler',\n",
       "                                                                   StandardScaler(c...\n",
       "                                                                                 fill_value='nan',\n",
       "                                                                                 missing_values=nan,\n",
       "                                                                                 strategy='constant',\n",
       "                                                                                 verbose=0)),\n",
       "                                                                  ('OHO',\n",
       "                                                                   OneHotEncoder(categories='auto',\n",
       "                                                                                 drop=None,\n",
       "                                                                                 dtype=<class 'numpy.float64'>,\n",
       "                                                                                 handle_unknown='ignore',\n",
       "                                                                                 sparse=False))],\n",
       "                                                           verbose=False),\n",
       "                                                  ['Rated',\n",
       "                                                   'Language_binary'])],\n",
       "                                   verbose=False)),\n",
       "                ('linear_model',\n",
       "                 LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,\n",
       "                                  normalize=False))],\n",
       "         verbose=False)"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer.pipeline.fit(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "b93cbcaf",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'pipeline' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-90-dcd0ce2243f3>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mpipeline\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'pipeline' is not defined"
     ]
    }
   ],
   "source": [
    "pipeline.predict(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "92e88659",
   "metadata": {},
   "outputs": [],
   "source": [
    "# testing the num_transformer\n",
    "num_transformer = Pipeline([\n",
    "    ('imputer', SimpleImputer(fill_value ='nan')),\n",
    "    ('scaler', StandardScaler())])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "8e4c9890",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.80235407, -1.21653733],\n",
       "       [ 0.43597955,  0.39506129],\n",
       "       [ 0.43597955, -0.32120476],\n",
       "       ...,\n",
       "       [ 0.68022923, -0.18690488],\n",
       "       [ 0.80235407,  0.17122815],\n",
       "       [ 0.80235407, -0.27643813]])"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "num_transformer.fit_transform(X[[\"Year\", \"Runtime\"]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "6a254702",
   "metadata": {},
   "outputs": [],
   "source": [
    "# testing the cat_transformer\n",
    "cat_transformer = Pipeline([\n",
    "    ('imputer', SimpleImputer(fill_value ='nan', strategy='constant')),\n",
    "    ('OHO', OneHotEncoder(handle_unknown='ignore', sparse = False))])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "733bbb46",
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_sparse = cat_transformer.fit_transform(X[[\"Rated\", \"Language_binary\"]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "c2c4f41b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10386, 28)"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_sparse.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "aaacb431",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Year</th>\n",
       "      <th>Rated</th>\n",
       "      <th>Language_binary</th>\n",
       "      <th>Runtime</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2003</td>\n",
       "      <td>NaN</td>\n",
       "      <td>English Available</td>\n",
       "      <td>50.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1997</td>\n",
       "      <td>R</td>\n",
       "      <td>English Available</td>\n",
       "      <td>122.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1997</td>\n",
       "      <td>Not Rated</td>\n",
       "      <td>English Available</td>\n",
       "      <td>90.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1992</td>\n",
       "      <td>NaN</td>\n",
       "      <td>English N/A</td>\n",
       "      <td>83.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1947</td>\n",
       "      <td>Passed</td>\n",
       "      <td>English Available</td>\n",
       "      <td>87.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10381</th>\n",
       "      <td>1978</td>\n",
       "      <td>PG</td>\n",
       "      <td>English Available</td>\n",
       "      <td>92.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10382</th>\n",
       "      <td>1998</td>\n",
       "      <td>R</td>\n",
       "      <td>English Available</td>\n",
       "      <td>123.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10383</th>\n",
       "      <td>2001</td>\n",
       "      <td>PG-13</td>\n",
       "      <td>English Available</td>\n",
       "      <td>96.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10384</th>\n",
       "      <td>2003</td>\n",
       "      <td>PG-13</td>\n",
       "      <td>English Available</td>\n",
       "      <td>112.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10385</th>\n",
       "      <td>2003</td>\n",
       "      <td>R</td>\n",
       "      <td>English Available</td>\n",
       "      <td>92.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10386 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       Year      Rated    Language_binary  Runtime\n",
       "0      2003        NaN  English Available     50.0\n",
       "1      1997          R  English Available    122.0\n",
       "2      1997  Not Rated  English Available     90.0\n",
       "3      1992        NaN        English N/A     83.0\n",
       "4      1947     Passed  English Available     87.0\n",
       "...     ...        ...                ...      ...\n",
       "10381  1978         PG  English Available     92.0\n",
       "10382  1998          R  English Available    123.0\n",
       "10383  2001      PG-13  English Available     96.0\n",
       "10384  2003      PG-13  English Available    112.0\n",
       "10385  2003          R  English Available     92.0\n",
       "\n",
       "[10386 rows x 4 columns]"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "0d29078e",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'preprocessor' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-97-39691ed0e2dc>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpreprocessor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit_transform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m5\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'preprocessor' is not defined"
     ]
    }
   ],
   "source": [
    "pd.DataFrame(preprocessor.fit_transform(X)).head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "b44c73c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#preprocessor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "25021f23",
   "metadata": {},
   "outputs": [],
   "source": [
    "def run(self):\n",
    "        self.set_pipeline()\n",
    "        self.pipeline.fit(self.X, self.y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "a1b8acab",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec050658",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "0a1ced64",
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate(self, X_test, y_test):\n",
    "        \"\"\"evaluates the pipeline on df_test and return the RMSE\"\"\"\n",
    "        y_pred = self.pipeline.predict(X_test)\n",
    "        rmse = compute_rmse(y_pred, y_test)\n",
    "        return round(rmse, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "81afb6ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_model(self):\n",
    "        \"\"\"Save the model into a .joblib format\"\"\"\n",
    "        joblib.dump(self.pipeline, 'model.joblib')\n",
    "        print(colored(\"model.joblib saved locally\", \"green\"))\n",
    "\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6420290d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "9d072878",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.4996588291326279"
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_dummy_pred = np.full(shape = (10386, 1),fill_value = y.mean())\n",
    "actual = df.avg_review_score\n",
    "predicted = y_dummy_pred\n",
    "\n",
    "mse = mean_squared_error(actual, predicted)\n",
    "rmse = math.sqrt(mse)\n",
    "\n",
    "rmse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46d069b1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.6"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
