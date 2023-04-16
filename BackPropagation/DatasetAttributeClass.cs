using CsvHelper.Configuration.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace BackPropagation
{
    public class DatasetAttributeClass
    {
        //[Name("ph")]
        public float in1 { get; set; }

        //Name("Hardness")]
        public float in2 { get; set; }

        //[Name("Solids")]
        public float in3 { get; set; }

        //[Name("Chloramines")]
        public float in4 { get; set; }

        //[Name("Sulfate")]
        public float in5 { get; set; }

        //[Name("Conductivity")]
        public float in6 { get; set; }
        public float in7 { get; set; }
        public float in8 { get; set; }
        public float in9 { get; set; }

        //[Name("Organic_carbon")]
        //public float in7 { get; set; }

        //[Name("Trihalomethanes")]
        //public float in8 { get; set; }

        //[Name("Turbidity")]
        //public float in9 { get; set; }

        // [Name("Potability")]
        public float out1 { get; set; }
    }
}
