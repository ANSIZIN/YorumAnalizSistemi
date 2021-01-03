using System;
using System.Windows.Forms;
using System.IO;
using HtmlAgilityPack;
using JR.Utils.GUI.Forms;
using Microsoft.ML.Data;
using Microsoft.ML;
using System.Data;
using System.Drawing;
using Microsoft.ML.Trainers;

namespace UzmanSistemlerProje
{
    public partial class Form1 : Form
    {        
        static string[] cekilenUrl = new string[6000];//çekilen urllerin tutulduğu dizi
        int y = 0; //Urlleri tutmak için sayaç
        static string[] yorumlar = new string[6000];//çekilen yorumların tutulduğu dizi
        int i = 0; //Yorumları tutmak için sayaç

        //static readonly string _path = "..\\..\\..\\Data\\YorumlarDataSet.csv";// Verinin alınacağı yol
        static readonly string _path = "C:\\Program Files\\Default Company Name\\Yorum Analiz Sistemi Setup\\Data\\YorumlarDataSet.csv";
        dynamic predictor; // Model tahmini için kullanılan dinamik değişken.      
        
        public Form1()
        {
            InitializeComponent();
            // Veri seti oluşturmak için kullanılan listview1 ayarları
            listView1.View = View.Details;
            listView1.Columns.Add("Url", 250);
            listView1.Columns.Add("Yorum",725);
            listView1.FullRowSelect = true;
            listView1.ShowItemToolTips = true;

            // Uygulamayı sitede test etmek için kullanılan listview2 ayarları
            listView2.View = View.Details;
            listView2.Columns.Add("Url", 250);
            listView2.Columns.Add("Yorum", 770);
            listView2.Columns.Add("Durum", 100);
            listView2.Columns.Add("Skor", 70);
            listView2.FullRowSelect = true;
            listView2.ShowItemToolTips = true;
        }

        // Ana Form yüklendikten sonra yapılacak işlemler
        private void Form1_Shown(object sender, EventArgs e)
        {
            EgitimYap();
            button3.Visible = true;
            button4.Visible = true;
            button5.Visible = true;
            button7.Visible = true;
        }

        // Oluşturulan Veri Setimiz ile modelin eğitilmesi
        public void EgitimYap()
        {
            label22.Text = "Model eğitiliyor...";
            //ML.net ile yeni bir “context” oluşturuyoruz.
            var context = new MLContext(seed: 0);

            // Girilen yolda bulunan veri setimizden veriler yükleniyor.Ayırıcı karakter olarak virgül kullanılıyor.
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            // Veri setimizi %20 test %80 eğitim olacak şekilde bölüyoruz.
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Eğitim için gerekli gördüğümüz özellikleri tanımlıyoruz.
            var options = new SdcaLogisticRegressionBinaryTrainer.Options()
            {
                // Yakınsama toleransını ayarlar.
                ConvergenceTolerance = 0.05f,
                // Eğitim verileri üzerinden maksimum iterasyon sayısını belirler.
                MaximumNumberOfIterations = 100000,
                // Pozitif sınıfın örneklerine biraz daha fazla ağırlık verir.
                //PositiveInstanceWeight = 1.2f,
            };

            // Model oluşturulur ve lojistik regresyon kullanılarak eğitilir.
            var pipeline = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Text")
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(options));
            var model = pipeline.Fit(trainData);

            // Modelin sonuçları test verisi üzerinden analiz edilir ve çıkartılır.
            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            var TP = metrics.ConfusionMatrix.Counts[0][0];
            var FP = metrics.ConfusionMatrix.Counts[1][0];
            var FN = metrics.ConfusionMatrix.Counts[0][1];
            var TN = metrics.ConfusionMatrix.Counts[1][1];


            var Prevalence = (TP + FN) / (TP + FP + FN + TN);
            var Accuracy = (TP + TN) / (TP + FP + FN + TN);
            var Auc = metrics.AreaUnderPrecisionRecallCurve;


            var Ppv = TP / (TP + FP); // Positive predictive value (PPV), Precision 
            var Fdr = FP / (TP + FP); // False discovery rate (FDR) 
            var For = FN / (FN + TN); // False omission rate (FOR) 
            var Npv = TN / (FN + TN); // Negative predictive value (NPV) 

            var Tpr = TP / (TP + FN); // True positive rate (TPR), Recall, Sensitivity, probability of detection, Power
            var Fpr = FP / (FP + TN); // False positive rate (FPR), Fall-out, probability of false alarm (1-Specificity)
            var Fnr = FN / (TP + FN); // False negative rate (FNR), Miss rate
            var Tnr = TN / (FP + TN); // True negative rate (TNR), Specificity (SPC), Selectivity

            var LrArti = (Tpr) / (Fpr); // Positive likelihood ratio (LR+)
            var LrEksi = (Fnr) / (Tnr); // Negative likelihood ratio (LR−)
            var Dor = (LrArti) / (LrEksi); // Diagnostic odds ratio (DOR)
            var F1 = 2 * ((Ppv * Tpr) / (Ppv + Tpr)); // F1 score


            CreateConfusionMatrix(TN, FP, FN, TP); //Confusion Matrix oluşturmak için fonksiyon
            CreateResults(Prevalence, Accuracy, Auc, Ppv, Fdr, For, Npv, Tpr, Fpr, Fnr, Tnr, LrArti, LrEksi, Dor, F1); //Sonuçları göstermek için fonksiyon

            // Tahmin yapmak için modeli kullanmak predictor değişkenine atıyoruz.
            predictor = context.Model.CreatePredictionEngine<Input, Output>(model);
            label22.Text = "Model eğitimi tamamlandı."; //Eğitim tamamlandığını ekranda gösteriyoruz.
        }

        //Confusion Matrixi tablo halinde ekrana yazdıran fonksiyon
        public void CreateConfusionMatrix(dynamic TN, dynamic FP, dynamic FN, dynamic TP)
        {
            DataTable table = new DataTable(); //tablomuzu oluşturduk..
            table.Columns.Add("Actual\nPredicted"); //table isimli tabloya ilk kolonumuzu ekledik…
            table.Columns.Add("Actual\nTrue"); //table isimli tabloya ikinci kolonumuzu ekledik…
            table.Columns.Add("Actual\nFalse"); //table isimli tabloya üçüncü kolonumuzu ekledik…

            DataRow row = table.NewRow(); //Tablo için satır oluşturduk.
            row["Actual\nPredicted"] = "Predicted True"; //1. satır Adı
            row["Actual\nTrue"] = TP; //1.değeri..
            row["Actual\nFalse"] = FP; //1. değeri..
            table.Rows.Add(row); //Satırı tabloya ekledik.


            DataRow row2 = table.NewRow(); //Tablo için 2. satır oluşturduk.
            row2["Actual\nPredicted"] = "Predicted False"; //2. satır Adı 
            row2["Actual\nTrue"] = FN; //2. satır değeri.
            row2["Actual\nFalse"] = TN; //2. satır değeri.
            table.Rows.Add(row2); //Satırı tabloya ekledik.

            dataGridView1.DataSource = table; //Tablomuzu görebilmek için gridControl’e yükledik..            
            dataGridView1.Columns[1].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);//tabloda kullanılan yazı sitili
            dataGridView1.Columns[2].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);//tabloda kullanılan yazı sitili
            dataGridView1.Columns[0].Width = 120;//tabloda kullanılan sütun genişliği
            dataGridView1.Columns[1].Width = 120;//tabloda kullanılan sütun genişliği
            dataGridView1.Columns[2].Width = 120;//tabloda kullanılan sütun genişliği

            //tablonun renklendirilmesi için kullanılan değişkenler
            DataGridViewCellStyle style = new DataGridViewCellStyle();
            style.ForeColor = Color.Red;
            dataGridView1.Rows[0].Cells[2].Style = style;
            dataGridView1.Rows[1].Cells[1].Style = style;

            DataGridViewCellStyle style1 = new DataGridViewCellStyle();
            style1.ForeColor = Color.Green;
            dataGridView1.Rows[0].Cells[1].Style = style1;
            dataGridView1.Rows[1].Cells[2].Style = style1;

        }

        // Performans metriklerinin tabnlo halinde ekrana yazdıran fonksiyon
        public void CreateResults(dynamic Prevalence, dynamic Accuracy, dynamic Auc, dynamic Ppv, dynamic Fdr, dynamic For, dynamic Npv, dynamic Tpr, dynamic Fpr, dynamic Fnr, dynamic Tnr, dynamic LrArti, dynamic LrEksi, dynamic Dor, dynamic F1)
        {
            DataTable table = new DataTable();
            table.Columns.Add("PREVALENCE");
            table.Columns.Add("ACCURACY");
            table.Columns.Add("AUC");
            table.Columns.Add("PPV");
            table.Columns.Add("FDR");
            table.Columns.Add("FOR");
            table.Columns.Add("NPV");
            table.Columns.Add("TPR");
            table.Columns.Add("FPR");
            table.Columns.Add("FNR");
            table.Columns.Add("TNR");

            DataRow row = table.NewRow();
            row["PREVALENCE"] = Prevalence.ToString("N2");
            row["ACCURACY"] = Accuracy.ToString("N2");
            row["AUC"] = Auc.ToString("N2");
            row["PPV"] = Ppv.ToString("N2");
            row["FDR"] = Fdr.ToString("N2");
            row["FOR"] = For.ToString("N2");
            row["NPV"] = Npv.ToString("N2");
            row["TPR"] = Tpr.ToString("N2");
            row["FPR"] = Fpr.ToString("N2");
            row["FNR"] = Fnr.ToString("N2");
            row["TNR"] = Tnr.ToString("N2");
            table.Rows.Add(row);

            dataGridView2.DataSource = table; //Tablomuzu görebilmek için gridControl’e yükledik.. 
            dataGridView2.Columns[0].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[1].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[2].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[3].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[4].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[5].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[6].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[7].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[8].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[9].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView2.Columns[10].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);




            DataTable table1 = new DataTable();
            table1.Columns.Add("LR+");
            table1.Columns.Add("LR-");
            table1.Columns.Add("DOR");
            table1.Columns.Add("F1");

            DataRow row1 = table1.NewRow();
            row1["LR+"] = LrArti.ToString("N2");
            row1["LR-"] = LrEksi.ToString("N2");
            row1["DOR"] = Dor.ToString("N2");
            row1["F1"] = F1.ToString("N2");
            table1.Rows.Add(row1);

            dataGridView3.DataSource = table1; //Tablomuzu görebilmek için gridControl’e yükledik.. 
            dataGridView3.Columns[0].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView3.Columns[1].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView3.Columns[2].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);
            dataGridView3.Columns[3].DefaultCellStyle.Font = new Font("Verdana", 14, FontStyle.Bold);

            dataGridView3.Columns[0].Width = 120;
            dataGridView3.Columns[1].Width = 120;
            dataGridView3.Columns[2].Width = 120;
            dataGridView3.Columns[3].Width = 120;
        }



        //Paneller arası geçişte diziler,sayaçlar,textboxlar ve listviewler temizlenmesi için fonksiyon
        public void YorumUrlDiziTemizle()
        {
            Array.Clear(cekilenUrl, 0, 6000);
            Array.Clear(yorumlar, 0, 6000);
            y = 0;
            i = 0;
            listView1.Items.Clear();
            listView2.Items.Clear();
            textBox1.Text = "";
            textBox3.Text = "";
        }

        //Veri Seti Oluşturma Panelini açmak için kullanılan buton
        private void button3_Click(object sender, EventArgs e)
        {
            YorumUrlDiziTemizle();
            panel2.Visible = false;
            panel3.Visible = false;
            panel4.Visible = false;
            panel1.Visible = true;

        }

        // Girilen urlden yorumları listview1'e aktaran buton
        private void button2_Click(object sender, EventArgs e)
        {
            if (textBox1.Text == "")
            {
                MessageBox.Show("Lütfen geçerli bir url girin");
            }
            else
            {
                try
                {
                    var web = new HtmlWeb();
                    var doc = web.Load(textBox1.Text);
                    var a = doc.DocumentNode.SelectNodes("//div[@class='comment-entry']/p/text()");
                    if (a != null)
                    {
                        if (y == 0)
                        {
                            cekilenUrl[y] = textBox1.Text;
                            y++;
                            foreach (var node in a)
                            {                               
                                yorumlar[i] = node.InnerText;
                                ListViewItem listitem = new ListViewItem(new[]
                                     {
                                         textBox1.Text,yorumlar[i]
                                     });

                                listView1.Items.Add(listitem);
                                i++;
                            }
                        }
                        else
                        {
                            int count = 0;


                            while (String.Compare(cekilenUrl[count], textBox1.Text) != 0 && cekilenUrl[count] != null)
                            {
                                count++;
                            }
                            if (count == y)
                            {
                                cekilenUrl[y] = textBox1.Text;
                                y++;
                                foreach (var node in a)
                                {
                                    //Console.WriteLine(node.InnerText + "\n");
                                    yorumlar[i] = node.InnerText;
                                    ListViewItem listitem = new ListViewItem(new[]
                                                       {
                                         textBox1.Text,yorumlar[i]
                                     });

                                    listView1.Items.Add(listitem);
                                    i++;
                                }
                            }
                            else
                            {
                                MessageBox.Show("Daha önce aynı Url kullanıldı.");
                            }
                        }

                    }
                    else
                    {
                        MessageBox.Show("Yanlış bir url girdiniz.");
                    }
                }
                catch
                {
                    MessageBox.Show("Girilen Url hatalı");
                }
            }
            MessageBox.Show("Çekilen toplam yorum sayısı:" + i.ToString());
        }

        // Yorumları metin dosyasına yazdırmak için kullanılan buton
        private void button1_Click(object sender, EventArgs e)
        {
            if (i == 0)
            {
                MessageBox.Show("Henüz yorum çekilmemiş.");
            }
            else
            {
                export2File(listView1, "   ");
            }
        }

        //Veri Seti Oluşturma Panelini kapatmak için kullanılan buton
        private void button6_Click(object sender, EventArgs e)
        {
            YorumUrlDiziTemizle();
            panel1.Visible = false;
        }

        //listview1'deki yorumları metin dosyasına kayıt eden fonksiyon
        private void export2File(ListView lv, string splitter)
        {
            string filename = "deneme";
            SaveFileDialog sfd = new SaveFileDialog();

            sfd.Title = "SaveFileDialog Export2File";
            sfd.Filter = "Text File (.txt) | *.txt";

            if (sfd.ShowDialog() == DialogResult.OK)
            {
                filename = sfd.FileName.ToString();
                if (filename != "")
                {
                    using (StreamWriter sw = new StreamWriter(filename))
                    {
                        foreach (ListViewItem item in lv.Items)
                        {                            
                            sw.WriteLine("{0}{1}{2}", ",\"", item.SubItems[1].Text,"\"");
                        }
                    }
                }
            }
        }

        // listview'de yorumların üzerine tıklayınca yorumları gösteren fonksiyon
        private void listView1_SelectedIndexChanged(object sender, EventArgs e)
        {
            ListView.SelectedIndexCollection indices = listView1.SelectedIndices;
            if (indices.Count > 0)
            {
                FlexibleMessageBox.Show(yorumlar[indices[0]]);
            }
        }
        
        // Eğitim Sonuçlarını göster butonu
        private void button7_Click(object sender, EventArgs e)
        {
            YorumUrlDiziTemizle();
            panel1.Visible = false;
            panel3.Visible = false;
            panel4.Visible = false;
            panel2.Visible = true;
        }
        
        // Eğitim Sonuçlarını kapat butonu
        private void button8_Click(object sender, EventArgs e)
        {
            panel2.Visible = false;
        }

        // Metin girerek test et panelini açar
        private void button4_Click(object sender, EventArgs e)
        {
            YorumUrlDiziTemizle();
            panel1.Visible = false;
            panel2.Visible = false;
            panel4.Visible = false;
            panel3.Visible = true;
        }

        // Girilen metinin sonuçlarını test etmek için kullanılan buton
        private void button9_Click(object sender, EventArgs e)
        {
            if (textBox2.Text == "")
            {
                FlexibleMessageBox.FONT = new Font("Verdana", 14, FontStyle.Bold);
                FlexibleMessageBox.Show("Lütfen bir metin giriniz.");
            }
            else
            {
                SonucDon(predictor);
            }
        }

        // Girilen metinin pozitif veya negatif olduğunu ve skorunu mesaj kutusu ile gösteren fonksiyon
        public void SonucDon(dynamic predictor)
        {
            var input = new Input { Text = textBox2.Text };
            var prediction = predictor.Predict(input);
            FlexibleMessageBox.FONT = new Font("Verdana", 14, FontStyle.Bold);
            FlexibleMessageBox.Show("Girilen Metin: " + (Convert.ToBoolean(prediction.Prediction) ? "Pozitif" : "Negatif") + "\n" + "Negatif Skoru: " + prediction.Probability, "Sonuç Mesajı");
        }

        // Metin girerek test et panelini kapatır.
        private void button10_Click(object sender, EventArgs e)
        {
            panel3.Visible = false;
        }

        // Siteden yorum çekerek test et panelini açar
        private void button5_Click(object sender, EventArgs e)
        {
            YorumUrlDiziTemizle();
            panel1.Visible = false;
            panel2.Visible = false;
            panel3.Visible = false;
            panel4.Visible = true;
        }

        // Siteden alınan yorumları listview2'ye yazdırır.Yorumun Pozitif veya Negatif olduğunu ve skorunuda listview2'ye yazdırır.
        private void button13_Click(object sender, EventArgs e)
        {
            if (textBox3.Text == "")
            {
                MessageBox.Show("Lütfen geçerli bir url girin");
            }
            else
            {
                try
                {
                    var web = new HtmlWeb();
                    var doc = web.Load(textBox3.Text);
                    var a = doc.DocumentNode.SelectNodes("//div[@class='comment-entry']/p/text()");
                    if (a != null)
                    {
                        if (y == 0)
                        {
                            cekilenUrl[y] = textBox3.Text;
                            y++;
                            foreach (var node in a)
                            {
                                yorumlar[i] = node.InnerText;                                
                                var input = new Input { Text = node.InnerText };
                                var prediction = predictor.Predict(input);

                                double skor = Cast(prediction.Probability, typeof(double));

                                String durum;
                                if ((Convert.ToBoolean(prediction.Prediction) ? "Pozitif" : "Negatif") == "Pozitif")
                                {
                                    durum = "Pozitif";
                                }
                                else
                                {
                                    durum = "Negatif";
                                }
                                ListViewItem listitem1 = new ListViewItem(new[]
                                         {
                                         textBox3.Text,node.InnerText,durum,skor.ToString("N2")
                                     });
                                listView2.Items.Add(listitem1);
                                i++;
                            }
                            colorListcolor(listView2);
                        }
                        else
                        {
                            int count = 0;
                            while (String.Compare(cekilenUrl[count], textBox3.Text) != 0 && cekilenUrl[count] != null)
                            {
                                count++;
                            }
                            if (count == y)
                            {
                                cekilenUrl[y] = textBox3.Text;
                                y++;
                                foreach (var node in a)
                                {
                                    yorumlar[i] = node.InnerText;
                                    var input = new Input { Text = node.InnerText };
                                    var prediction = predictor.Predict(input);

                                    double skor = Cast(prediction.Probability, typeof(double));

                                    String durum;
                                    if ((Convert.ToBoolean(prediction.Prediction) ? "Pozitif" : "Negatif") == "Pozitif")
                                    {
                                        durum = "Pozitif";
                                    }
                                    else
                                    {
                                        durum = "Negatif";
                                    }
                                    ListViewItem listitem1 = new ListViewItem(new[]
                                             {
                                         textBox3.Text,node.InnerText,durum,skor.ToString("N2")
                                     });
                                    listView2.Items.Add(listitem1);
                                    i++;
                                }
                                colorListcolor(listView2);
                            }
                            else
                            {
                                MessageBox.Show("Daha önce aynı Url kullanıldı.");
                            }
                        }
                    }
                    else
                    {
                        MessageBox.Show("Yanlış bir url girdiniz.");
                    }
                }
                catch
                {
                    MessageBox.Show("Girilen Url hatalı");
                }
            }
            MessageBox.Show("Çekilen toplam yorum sayısı:" + listView2.Items.Count.ToString());
        }

        // Yorumun Pozitif veya Negatif olmasına göre listview2'de renklendirme yapılır.
        public static void colorListcolor(ListView lsvMain)
        {
            foreach (ListViewItem lvw in lsvMain.Items)
            {
                lvw.UseItemStyleForSubItems = false;

                for (int i = 0; i < lsvMain.Columns.Count; i++)
                {
                    if (lvw.SubItems[2].Text.ToString() == "Negatif")
                    {
                        lvw.SubItems[2].BackColor = Color.Red;
                        lvw.SubItems[2].ForeColor = Color.White;
                    }
                    else
                    {
                        lvw.SubItems[2].BackColor = Color.Green;
                        lvw.SubItems[2].ForeColor = Color.White;
                    }
                }
            }
        }

        // Model skoru dynamic veri tipinden istenilen veri tipine dönüştürülür.
        public static dynamic Cast(dynamic obj, Type castTo)
        {
            return Convert.ChangeType(obj, castTo);
        }

        // listview2'de yorumların üzerine tıklayınca yorumlar geniş halde mesaj kutusunda görünür.
        private void listView2_SelectedIndexChanged(object sender, EventArgs e)
        {
            ListView.SelectedIndexCollection indices = listView2.SelectedIndices;
            if (indices.Count > 0)
            {
                FlexibleMessageBox.Show(yorumlar[indices[0]]);
            }
        }

        // Siteden yorum çekerek test et panelini kapatır.
        private void button11_Click(object sender, EventArgs e)
        {
            YorumUrlDiziTemizle();
            panel4.Visible = false;
        }
    }

    //Modelin girdi kısmı için oluşturulan sınıf
    public class Input
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool IsNegative;

        [LoadColumn(1)]
        public string Text;
    }

    //Modelin çıktı kısmı için oluşturulan ınıf
    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}
